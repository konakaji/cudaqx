# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from torch.func import functional_call
from cudaq import spin
import cudaq
from cudaq_solvers.gqe_algorithm.factory import Factory
from cudaq_solvers.gqe_algorithm.gqe import get_default_config
from cudaq_solvers.gqe_algorithm.loss import GRPOLoss
from cudaq_solvers.gqe_algorithm.pipeline import Pipeline
from cudaq_solvers.gqe_algorithm.scheduler import DefaultScheduler, CosineScheduler, VarBasedScheduler
from cudaq_solvers.gqe_algorithm.utils import get_gqe_pauli_pool
import cudaq_solvers as solvers


def _torch_cuda_kernel_supported():
    """Check if the installed PyTorch can execute CUDA kernels on this GPU."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        torch.zeros(1, device='cuda')
        return True, ""
    except RuntimeError:
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            raise
        name = torch.cuda.get_device_name()
        return False, (f"PyTorch CUDA wheel does not include kernels for "
                       f"{name} ({arch})")


_cuda_supported, _cuda_skip_reason = _torch_cuda_kernel_supported()
requires_cuda_kernels = pytest.mark.skipif(not _cuda_supported,
                                           reason=_cuda_skip_reason)

qubit_count = 2
# Define a simple Hamiltonian: Z₀ + Z₁
ham = spin.z(0) + spin.z(1)


# Generate an operator pool for the GQE
def ops_pool(n):
    pool = []
    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))
        pool.append(cudaq.SpinOperator(spin.y(i)))
        pool.append(cudaq.SpinOperator(spin.z(i)))
    for i in range(n - 1):
        pool.append(cudaq.SpinOperator(spin.z(i) *
                                       spin.z(i + 1)))  # ZZ entangling
    return pool


pool = ops_pool(qubit_count)


# Helper functions to extract coeffs and Pauli words
def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(qubit_count) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(qcount: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(qcount)
    h(q)
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


# Global cost function for GQE
def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    full_coeffs = []
    full_words = []
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    return cudaq.observe(kernel, ham, qubit_count, full_coeffs,
                         full_words).expectation()


def test_default_scheduler():
    """Test the DefaultScheduler temperature scheduling"""
    scheduler = DefaultScheduler(start=1.0, delta=0.1)
    assert scheduler.get_inverse_temperature() == 1.0
    scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 1.1, atol=1e-6)
    for _ in range(9):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 2.0, atol=1e-6)


def test_cosine_scheduler():
    """Test the CosineScheduler temperature scheduling"""
    scheduler = CosineScheduler(minimum=1.0, maximum=5.0, frequency=10)
    # Initial temperature should be at midpoint
    assert np.isclose(scheduler.get_inverse_temperature(), 3.0, atol=1e-6)

    # After 5 updates, should be at maximum (cos(π)=-1)
    for _ in range(5):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 5.0, atol=1e-6)

    # After 10 updates total, should be back near starting point (cos(2π)=1)
    for _ in range(5):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 1.0, atol=1e-6)


def test_variance_scheduler():
    """Test the VarBasedScheduler temperature scheduling"""
    import torch
    scheduler = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.1)

    # Test initial temperature
    assert scheduler.get_inverse_temperature() == 2.0

    # Simulate high variance scenario (should increase temperature)
    high_var_energies = torch.tensor([1.0, 5.0, 2.0, 6.0, 3.0])  # var ≈ 3.5
    initial_temp = scheduler.current_temperature
    scheduler.update(energies=high_var_energies)
    temp_after_high_var = scheduler.current_temperature
    assert temp_after_high_var > initial_temp  # Temperature should increase

    # Simulate low variance scenario (should decrease temperature)
    scheduler2 = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.5)
    low_var_energies = torch.tensor([1.0, 1.1, 1.05, 0.95, 1.02])  # var ≈ 0.003
    initial_temp2 = scheduler2.current_temperature
    scheduler2.update(energies=low_var_energies)
    temp_after_low_var = scheduler2.current_temperature
    assert temp_after_low_var < initial_temp2  # Temperature should decrease

    # Test minimum temperature bound
    scheduler3 = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.1)
    for _ in range(100):  # Many decreases
        scheduler3.update(energies=low_var_energies)
    final_temp = scheduler3.current_temperature
    assert final_temp >= 0.01  # Should not go below min_temp (0.01)


class _ToyPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 3)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.register_buffer("logit_bias", torch.tensor([0.0, 0.1, 0.2]))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3))

    def forward(self, idx):
        logits = self.dropout(self.embedding(idx)) + self.logit_bias
        return SimpleNamespace(logits=logits)


def _make_test_pipeline():
    cfg = get_default_config()
    cfg.num_samples = 2
    cfg.ngates = 2
    cfg.buffer_size = 4
    cfg.warmup_size = 2
    cfg.batch_size = 2
    cfg.step_per_epoch = 2
    return Pipeline(cfg, None, [], _ToyPolicy(), Factory())


def test_reference_policy_is_frozen_and_refreshed_each_epoch(monkeypatch):
    pipeline = _make_test_pipeline()
    state_seen_during_rollout = []
    state_dict_keys = set(pipeline.state_dict())
    rollout_beta = pipeline.scheduler.get_inverse_temperature()

    def record_rollout():
        state_seen_during_rollout.append({
            name: tensor.detach().clone()
            for name, tensor in pipeline._reference_state.items()
        })
        pipeline.scheduler.update()

    monkeypatch.setattr(pipeline, "collect_rollout", record_rollout)

    with torch.no_grad():
        pipeline.model.embedding.weight.add_(1.0)
    pipeline.train()
    pipeline.on_train_epoch_start()

    current_parameter = next(pipeline.model.parameters())
    reference_parameter = pipeline._reference_state["embedding.weight"]
    reference_buffer = pipeline._reference_state["logit_bias"]
    torch.testing.assert_close(reference_parameter, current_parameter)
    torch.testing.assert_close(reference_buffer, pipeline.model.logit_bias)
    assert reference_parameter.data_ptr() != current_parameter.data_ptr()
    assert reference_buffer.data_ptr() != pipeline.model.logit_bias.data_ptr()
    assert not reference_parameter.requires_grad
    torch.testing.assert_close(state_seen_during_rollout[0]["embedding.weight"],
                               current_parameter)
    assert pipeline._reference_inverse_temperature == rollout_beta
    assert pipeline.scheduler.get_inverse_temperature() != rollout_beta
    assert set(pipeline.state_dict()) == state_dict_keys

    frozen_parameter = reference_parameter.detach().clone()
    frozen_buffer = reference_buffer.detach().clone()
    with torch.no_grad():
        current_parameter.add_(1.0)
        pipeline.model.logit_bias.add_(1.0)
    torch.testing.assert_close(reference_parameter, frozen_parameter)
    torch.testing.assert_close(reference_buffer, frozen_buffer)

    reference_idx = torch.tensor([[0, 1, 2], [0, 2, 1]])
    first_reference = pipeline._get_reference_log_probs(reference_idx)
    second_reference = pipeline._get_reference_log_probs(reference_idx)
    torch.testing.assert_close(first_reference, second_reference)
    assert pipeline.model.training

    next_rollout_beta = pipeline.scheduler.get_inverse_temperature()
    pipeline.on_train_epoch_start()
    torch.testing.assert_close(pipeline._reference_state["embedding.weight"],
                               current_parameter)
    torch.testing.assert_close(pipeline._reference_state["logit_bias"],
                               pipeline.model.logit_bias)
    assert pipeline._reference_inverse_temperature == next_rollout_beta


def test_old_log_probs_are_recomputed_for_each_batch(monkeypatch):
    pipeline = _make_test_pipeline()
    monkeypatch.setattr(pipeline, "collect_rollout", pipeline.scheduler.update)
    monkeypatch.setattr(pipeline, "log_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "log", lambda *args, **kwargs: None)
    pipeline.on_train_epoch_start()
    reference_beta = pipeline._reference_inverse_temperature
    assert pipeline.scheduler.get_inverse_temperature() != reference_beta

    with torch.no_grad():
        pipeline.model.embedding.weight.add_(0.5)

    first_idx = torch.tensor([[0, 1, 2], [0, 2, 1]])
    second_idx = torch.tensor([[0, 0, 1], [0, 1, 0]])
    first_batch = {"idx": first_idx, "energy": torch.tensor([0.0, 1.0])}
    second_batch = {"idx": second_idx, "energy": torch.tensor([2.0, 4.0])}

    seen_old_log_probs = []
    seen_old_log_probs_requires_grad = []
    seen_inverse_temperatures = []
    original_compute = pipeline.loss.compute

    def record_compute(*args, **kwargs):
        seen_old_log_probs_requires_grad.append(
            kwargs["old_log_probs"].requires_grad)
        seen_old_log_probs.append(kwargs["old_log_probs"].detach().clone())
        seen_inverse_temperatures.append(kwargs["inverse_temperature"])
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(pipeline.loss, "compute", record_compute)
    pipeline.training_step(first_batch, 0)
    pipeline.training_step(second_batch, 1)

    expected = []
    was_training = pipeline.model.training
    pipeline.model.eval()
    with torch.no_grad():
        for idx in (first_idx, second_idx):
            reference_logits = functional_call(pipeline.model,
                                               pipeline._reference_state,
                                               (idx,)).logits
            expected.append(
                pipeline.loss.log_prob(idx[:, 1:], reference_logits,
                                       reference_beta))
    pipeline.model.train(was_training)

    torch.testing.assert_close(seen_old_log_probs[0], expected[0])
    torch.testing.assert_close(seen_old_log_probs[1], expected[1])
    assert not torch.equal(expected[0], expected[1])
    assert seen_old_log_probs_requires_grad == [False, False]
    assert seen_inverse_temperatures == [reference_beta, reference_beta]


def test_grpo_ratio_at_unity_keeps_policy_gradient():
    loss_fn = GRPOLoss()
    gate_logits = torch.tensor([[[0.2, -0.1]], [[-0.3, 0.4]]],
                               requires_grad=True)
    gate_indices = torch.tensor([[0], [1]])
    energies = torch.tensor([0.0, 1.0])
    old_log_probs = loss_fn.log_prob(gate_indices, gate_logits, 1.0).detach()

    loss = loss_fn.compute(energies,
                           gate_logits,
                           gate_indices, {},
                           inverse_temperature=1.0,
                           old_log_probs=old_log_probs)
    loss.backward()

    assert gate_logits.grad[1].abs().sum() > 0


@requires_cuda_kernels
def test_solvers_gqe_basic():
    """Test basic GQE with config"""
    print("Setting up config...")
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 25
    cfg.ngates = 4
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = True
    cfg.save_dir = "./output/"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0  # Physical bound for simple Z₀ + Z₁ Hamiltonian


@requires_cuda_kernels
def test_solvers_gqe_small_transformer():
    """Test GQE with small transformer config"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = True
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


@requires_cuda_kernels
def test_solvers_gqe_with_gflow_loss():
    """Test GQE with GFlow loss function"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"
    cfg.loss = "gflow"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_exp_loss():
    """Test GQE with Exponential loss function"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"
    cfg.loss = "exp"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_variance_scheduler():
    """Test GQE with variance-based temperature scheduler"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.scheduler = 'variance'
    cfg.target_variance = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_cosine_scheduler():
    """Test GQE with cosine temperature scheduler"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.scheduler = 'cosine'
    cfg.temperature_min = 1.5
    cfg.temperature_max = 3.0
    cfg.scheduler_frequency = 20
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


@requires_cuda_kernels
def test_solvers_gqe_larger_molecule():
    """Test GQE with a larger number of gates"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 100
    cfg.ngates = 30
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_invalid_inputs():
    """Test error handling for invalid inputs"""
    cfg = get_default_config()

    # Test invalid number of samples
    cfg.num_samples = 0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)

    # Test invalid learning rate
    cfg.num_samples = 5
    cfg.lr = -1.0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)

    # Test invalid temperature
    cfg.lr = 1e-6
    cfg.temperature = -1.0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)


def test_get_gqe_pauli_pool():
    """Test GQE Pauli operator pool generation"""
    num_qubits = 4
    num_electrons = 2
    params = [0.01, -0.01, 0.05, -0.05]

    # Generate pool
    pool = get_gqe_pauli_pool(num_qubits, num_electrons, params)

    # Pool should be a list
    assert isinstance(pool, list)

    # Pool should not be empty
    assert len(pool) > 0

    # First operator should be identity
    identity_terms = list(pool[0])
    assert len(identity_terms) == 1
    pauli_word = identity_terms[0].get_pauli_word(num_qubits)
    assert pauli_word == "IIII"

    # All operators should be SpinOperators
    for op in pool:
        assert isinstance(op, cudaq.SpinOperator)

    # Check that pool contains parameterized operators
    # Pool size should be: 1 (identity) + (num_uccsd_terms * num_params)
    # At minimum, we expect more operators than just identity
    assert len(pool) > len(params)

    # Verify some operators have the expected parameter scaling
    non_identity_ops = pool[1:]  # Skip identity
    found_scaled_ops = False
    for op in non_identity_ops:
        terms = list(op)
        for term in terms:
            coeff = term.evaluate_coefficient()
            # Check if coefficient matches one of our params
            for param in params:
                if np.isclose(abs(coeff.real), abs(param), atol=1e-10):
                    found_scaled_ops = True
                    break
            if found_scaled_ops:
                break
        if found_scaled_ops:
            break

    assert found_scaled_ops, "Pool should contain operators scaled by the provided parameters"
