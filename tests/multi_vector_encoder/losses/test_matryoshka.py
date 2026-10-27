"""MultiVectorMatryoshkaLoss: repeat a wrapped multi-vector loss at several token embedding
dimensionalities, truncating and re-normalizing every token embedding per dimensionality.

The core contracts pinned here:
- with only the full dimension, the wrapper trains exactly like the wrapped loss (value and
  gradients) on a pipeline ending in Normalize, where the re-normalization is idempotent,
- the multi-dimension loss equals the manual weighted sum over truncated re-normalized inputs,
- the gradient-cached path (decorating ``calculate_loss``) matches the model-forward-patching path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder import losses as mve_losses

DIM = 8


class _PassthroughModel(nn.Module):
    """Stub model returning the ``token_embeddings`` already placed in the feature dict.

    Defined with a real ``forward`` (not ``__call__``) so the wrapper losses can patch
    ``model.forward`` like they do on a real MultiVectorEncoder. The wrapper and the wrapped loss
    must share one instance, as the wrapper patches the instance the wrapped loss calls.
    """

    def __init__(self, embedding_dim: int | None = DIM) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        return features

    def get_embedding_dimension(self) -> int | None:
        return self.embedding_dim


class _NormalizeModel(_PassthroughModel):
    """Stub mirroring the real pipeline's trailing Normalize module. With the loss-facing output
    L2-normalized per token, shrink's re-normalization at the full dimension is a gradient no-op
    (projecting twice equals projecting once), like on a real model."""

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        return {
            "token_embeddings": F.normalize(features["token_embeddings"], p=2, dim=-1),
            "attention_mask": features["attention_mask"],
        }


def _make_feature(t_tokens: int, batch: int, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    emb = F.normalize(torch.randn(batch, t_tokens, DIM, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    return {
        "token_embeddings": emb,
        "attention_mask": torch.ones(batch, t_tokens, dtype=torch.bool),
    }


def _build_features(batch: int = 4) -> list[dict[str, Tensor]]:
    return [
        _make_feature(t_tokens=6, batch=batch, seed=1),
        _make_feature(t_tokens=10, batch=batch, seed=2),
        _make_feature(t_tokens=14, batch=batch, seed=3),
    ]


def _leaves(features: list[dict[str, Tensor]]) -> list[Tensor]:
    # The wrapper replaces the dict entries with shrunk tensors, so keep the leaf references.
    return [sf["token_embeddings"] for sf in features]


def _shrink_reference(tensor: Tensor, dim: int) -> Tensor:
    return F.normalize(tensor[..., :dim], p=2, dim=-1)


def _mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model, **kwargs)


def _cached_mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=model, mini_batch_size=2, show_progress_bar=False, **kwargs
    )


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_full_dim_only_matches_wrapped_loss(build_inner) -> None:
    """With only the full dimensionality, the wrapper must reproduce the wrapped loss exactly,
    value and gradients both. The gradient equality relies on the model output being normalized
    (see _NormalizeModel), which is the standard multi-vector pipeline."""
    plain_features = _build_features()
    plain_value = build_inner(_NormalizeModel())(plain_features, None)
    plain_value.backward()

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _NormalizeModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, build_inner(model), matryoshka_dims=[DIM])
    value = loss(wrapped_features, None)
    value.backward()

    assert torch.allclose(value, plain_value, atol=1e-6)
    for leaf, plain_sf in zip(leaves, plain_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, plain_sf["token_embeddings"].grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("use_generator", [False, True])
def test_matryoshka_equals_manual_sum_of_truncated_losses(build_inner, use_generator: bool) -> None:
    """The multi-dimension wrapper must equal the weighted sum of the wrapped loss over manually
    truncated and re-normalized inputs. The single model forward is cached and shrunk per
    dimensionality, so this also pins the cache replay."""
    weights = [1.0, 0.5]

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(
        model, build_inner(model), matryoshka_dims=[DIM, DIM // 2], matryoshka_weights=weights
    )
    value = loss(iter(wrapped_features) if use_generator else wrapped_features, None)
    value.backward()

    manual_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    # The wrapper shrinks at the full dimension too (a no-op slice plus a re-normalization), so the
    # exact-mirror reference applies _shrink_reference at both dimensionalities.
    manual_value = 0.0
    for weight, dim in zip(weights, [DIM, DIM // 2]):
        truncated = [
            {
                "token_embeddings": _shrink_reference(sf["token_embeddings"], dim),
                "attention_mask": sf["attention_mask"],
            }
            for sf in manual_features
        ]
        manual_value = manual_value + weight * plain(truncated, None)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_cached_matryoshka_matches_non_cached() -> None:
    """The gradient-cached path decorates ``calculate_loss`` instead of patching the model forward.
    Both paths must produce the same loss and the same gradients."""
    dims = [DIM, DIM // 2, 2]
    weights = [1.0, 0.5, 0.25]

    plain_features = _build_features()
    plain_leaves = _leaves(plain_features)
    plain_model = _PassthroughModel()
    plain_loss = mve_losses.MultiVectorMatryoshkaLoss(
        plain_model, _mnrl(plain_model), matryoshka_dims=dims, matryoshka_weights=weights
    )
    plain_value = plain_loss(plain_features, None)
    plain_value.backward()

    cached_features = _build_features()
    cached_leaves = _leaves(cached_features)
    cached_model = _PassthroughModel()
    cached_loss = mve_losses.MultiVectorMatryoshkaLoss(
        cached_model, _cached_mnrl(cached_model), matryoshka_dims=dims, matryoshka_weights=weights
    )
    cached_value = cached_loss(cached_features, None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_leaf, cached_leaf in zip(plain_leaves, cached_leaves):
        assert plain_leaf.grad.abs().sum() > 0
        assert torch.allclose(plain_leaf.grad, cached_leaf.grad, atol=1e-6)


def test_matryoshka_does_not_truncate_kd_labels() -> None:
    """Teacher scores have shape (batch, N). Unlike the dense MatryoshkaLoss, which truncates
    embedding-shaped labels for distillation, multi-vector labels are always scores and must reach
    the wrapped loss untouched for every dimensionality."""
    labels = torch.tensor([[5.0, 1.0], [4.0, 0.5], [3.0, 0.2], [6.0, 2.0]])

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, kd, matryoshka_dims=[DIM, DIM // 2])
    value = loss(wrapped_features, labels)
    value.backward()

    manual_features = _build_features()
    plain = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    manual_value = 0.0
    for dim in [DIM, DIM // 2]:
        truncated = [
            {
                "token_embeddings": _shrink_reference(sf["token_embeddings"], dim),
                "attention_mask": sf["attention_mask"],
            }
            for sf in manual_features
        ]
        manual_value = manual_value + plain(truncated, labels)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_empty_dims_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="at least one dimension"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[])


def test_non_positive_dim_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="must be > 0"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[8, 0])


def test_weights_length_mismatch_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="same length"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[8, 4], matryoshka_weights=[1.0])


def test_dim_exceeding_model_dim_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="cannot exceed the model's embedding dimension of 8"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[16, 8])


def test_missing_full_dim_warns(caplog) -> None:
    model = _PassthroughModel()
    with caplog.at_level("WARNING"):
        mve_losses.MultiVectorMatryoshkaLoss(model, _mnrl(model), matryoshka_dims=[4, 2])
    assert "is not included in matryoshka_dims" in caplog.text


def test_dims_and_weights_sorted_descending() -> None:
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorMatryoshkaLoss(
        model, _mnrl(model), matryoshka_dims=[2, 8, 4], matryoshka_weights=[0.25, 1.0, 0.5]
    )
    assert loss.matryoshka_dims == (8, 4, 2)
    assert loss.matryoshka_weights == (1.0, 0.5, 0.25)
    assert loss.get_config_dict() == {
        "loss": "MultiVectorMultipleNegativesRankingLoss",
        "matryoshka_dims": (8, 4, 2),
        "matryoshka_weights": (1.0, 0.5, 0.25),
        "n_dims_per_step": -1,
    }


def test_matryoshka_on_a_real_model() -> None:
    """End to end on a real MultiVectorEncoder pipeline: the wrapper patches ``model.forward``, the
    losses pass ``task=`` kwargs through it, and gradients must reach the transformer."""
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.util import batch_to_device

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.train()
    dim = model.get_embedding_dimension()
    features = [
        batch_to_device(
            model.preprocess(["what is late interaction", "who wrote hamlet"], task="query"), model.device
        ),
        batch_to_device(
            model.preprocess(
                ["late interaction scores query and document tokens directly", "hamlet was written by shakespeare"],
                task="document",
            ),
            model.device,
        ),
    ]
    inner = mve_losses.MultiVectorMultipleNegativesRankingLoss(model)
    loss = mve_losses.MultiVectorMatryoshkaLoss(model, inner, matryoshka_dims=[dim, dim // 2])
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    grads = [p.grad for p in model[0].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert model.forward.__self__ is model, "the original model.forward must be restored"
