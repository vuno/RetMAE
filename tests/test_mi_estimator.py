"""Regression tests for the sampled MINE objective, independent of training dependencies."""

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/tasks/pretrain/backbone/mi_estimator.py"
SPEC = importlib.util.spec_from_file_location("mi_estimator", MODULE_PATH)
mi_estimator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mi_estimator)
MINE = mi_estimator.MINE


@pytest.mark.parametrize("token_target", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_matches_reference_value_and_gradients(token_target, batch_size):
    """Match CLUB's formula and gradients with identical replacement samples."""
    torch.manual_seed(10)
    model = MINE(x_dim=5, y_dim=3, hidden_dim=7).double()
    x = torch.randn(batch_size, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(batch_size, 3, dtype=torch.float64, requires_grad=True)
    targets = y.unsqueeze(1) if token_target else y

    torch.manual_seed(20)
    actual = model.learning_loss(x, targets)
    actual_grads = torch.autograd.grad(actual, (x, y, *model.parameters()))

    torch.manual_seed(20)
    index = torch.randint(batch_size, (batch_size,), device=y.device)
    projected = model.input_norm(model.input_proj(x))
    joint = model.F_func(torch.cat([projected, y], dim=-1))
    marginal = model.F_func(torch.cat([projected, y[index]], dim=-1))
    expected = -(joint.mean() - marginal.exp().mean().log())
    expected_grads = torch.autograd.grad(expected, (x, y, *model.parameters()))

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_critic_scores_remain_finite(dtype):
    """Keep loss and gradients finite when direct exponentiation would overflow."""
    model = MINE(x_dim=5, y_dim=3, hidden_dim=7).to(dtype=dtype)
    with torch.no_grad():
        model.F_func[-1].bias.fill_(1000)
    x = torch.randn(8, 5, dtype=dtype, requires_grad=True)
    y = torch.randn(8, 3, dtype=dtype, requires_grad=True)
    loss = model.learning_loss(x, y)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    for tensor in (x, y, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_critic_only_receives_linear_number_of_pairs():
    """Avoid the old quadratic pair tensor."""
    model = MINE(x_dim=5, y_dim=3, hidden_dim=7)
    pair_shapes = []

    def record_shape(module, inputs):
        pair_shapes.append(inputs[0].shape)

    hook = model.F_func.register_forward_pre_hook(record_shape)
    try:
        model(torch.randn(8, 5), torch.randn(8, 3))
    finally:
        hook.remove()
    assert pair_shapes == [torch.Size([8, 6]), torch.Size([8, 6])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_backward():
    """Sample negative indices on the target device during GPU training."""
    model = MINE(x_dim=5, y_dim=3, hidden_dim=7).cuda()
    x = torch.randn(8, 5, device="cuda", requires_grad=True)
    y = torch.randn(8, 3, device="cuda", requires_grad=True)
    loss = model.learning_loss(x, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()
