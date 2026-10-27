"""MultiVectorTokenPoolingLoss: repeat a wrapped multi-vector loss once per token pooling variant,
each time with that variant applied to the token embeddings and scoring masks.

The core contracts pinned here:
- a passthrough-only wrapper is exactly the wrapped loss (value and gradients),
- the multi-variant loss equals the manual weighted sum over manually pooled inputs, with queries
  left unpooled by the default document-only task,
- pooled terms move gradients into the underlying embeddings (through the cluster means),
- the gradient-cached path (decorating ``calculate_loss``) matches the model-forward-patching path,
- the two loss modifiers stack around one loss, cached included.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.multi_vector_encoder.modules import (
    BaseTokenPooling,
    HierarchicalTokenPooling,
    LambdaTokenPooling,
)

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
        self.forward_calls = 0

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        self.forward_calls += 1
        return features

    def get_embedding_dimension(self) -> int | None:
        return self.embedding_dim


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
    return [sf["token_embeddings"] for sf in features]


def _pool_reference(features: list[dict[str, Tensor]], pooling: BaseTokenPooling | None) -> list[dict[str, Tensor]]:
    """Pool each column the way a pipeline pooling module would at inference: per task, on top of
    the scoring mask. Fresh dicts, as the pooling forward rewrites the dict it is given."""
    if pooling is None:
        return [dict(sf) for sf in features]
    pooled = []
    for idx, sf in enumerate(features):
        task = "query" if idx == 0 else "document"
        pooled.append(
            pooling({"token_embeddings": sf["token_embeddings"], "attention_mask": sf["attention_mask"]}, task=task)
        )
    return pooled


def _mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model, **kwargs)


def _cached_mnrl(model: nn.Module, **kwargs) -> nn.Module:
    return mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=model, mini_batch_size=2, show_progress_bar=False, **kwargs
    )


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("use_none", [False, True])
def test_passthrough_only_matches_wrapped_loss(build_inner, use_none: bool) -> None:
    plain_features = _build_features()
    plain_value = build_inner(_PassthroughModel())(plain_features, None)
    plain_value.backward()

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    pooling = None if use_none else HierarchicalTokenPooling(pool_factor=1, normalize_embeddings=False)
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, build_inner(model), token_poolings=[pooling])
    value = loss(wrapped_features, None)
    value.backward()

    assert torch.allclose(value, plain_value, atol=1e-6)
    for leaf, plain_sf in zip(leaves, plain_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, plain_sf["token_embeddings"].grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_generator", [False, True])
def test_token_pooling_equals_manual_sum_of_pooled_losses(build_inner, reverse: bool, use_generator: bool) -> None:
    """The multi-variant wrapper must equal the weighted sum of the wrapped loss over manually
    pooled inputs, with the query column untouched (documents-only default task). The single model
    forward per column is cached and each variant pools the cached full-resolution output."""
    pooling = HierarchicalTokenPooling(pool_factor=2)
    variants = [(1.0, None), (0.5, pooling)]
    if reverse:
        variants.reverse()

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model,
        build_inner(model),
        token_poolings=[variant for _, variant in variants],
        weights=[weight for weight, _ in variants],
    )
    value = loss(iter(wrapped_features) if use_generator else wrapped_features, None)
    value.backward()
    if build_inner is _mnrl:
        assert model.forward_calls == len(wrapped_features)

    manual_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    manual_value = sum(weight * plain(_pool_reference(manual_features, variant), None) for weight, variant in variants)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert leaf.grad.abs().sum() > 0
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)

    with torch.no_grad():
        torch.testing.assert_close(loss(_build_features(), None), manual_value)


def test_pooled_terms_move_gradients() -> None:
    """With all weight on the pooled variant, gradients must still reach the input embeddings.
    The cluster assignment is a constant, gradients flow through the cluster means."""
    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model,
        _mnrl(model),
        token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)],
        weights=[0.0, 1.0],
    )
    value = loss(wrapped_features, None)
    value.backward()

    manual_features = _build_features()
    pooled_features = _pool_reference(manual_features, HierarchicalTokenPooling(pool_factor=2))
    pooled_value = _mnrl(_PassthroughModel())(pooled_features, None)
    pooled_value.backward()

    assert torch.allclose(value, pooled_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert leaf.grad.abs().sum() > 0, "the pooled loss term must move the underlying embeddings"
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_cached_token_pooling_matches_non_cached() -> None:
    """The gradient-cached path decorates ``calculate_loss`` and pools the cached embedding chunks
    (leaving pooled chunks of different widths to the loss's re-padding). Both paths must produce
    the same loss and the same gradients."""
    factors = [1, 2, 4]
    weights = [1.0, 0.5, 0.25]

    plain_features = _build_features()
    plain_leaves = _leaves(plain_features)
    plain_model = _PassthroughModel()
    plain_loss = mve_losses.MultiVectorTokenPoolingLoss(
        plain_model,
        _mnrl(plain_model),
        token_poolings=[HierarchicalTokenPooling(pool_factor=factor) for factor in factors],
        weights=weights,
    )
    plain_value = plain_loss(plain_features, None)
    plain_value.backward()

    cached_features = _build_features()
    cached_leaves = _leaves(cached_features)
    cached_model = _PassthroughModel()
    cached_loss = mve_losses.MultiVectorTokenPoolingLoss(
        cached_model,
        _cached_mnrl(cached_model),
        token_poolings=[HierarchicalTokenPooling(pool_factor=factor) for factor in factors],
        weights=weights,
    )
    cached_value = cached_loss(cached_features, None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_leaf, cached_leaf in zip(plain_leaves, cached_leaves):
        assert plain_leaf.grad.abs().sum() > 0
        assert torch.allclose(plain_leaf.grad, cached_leaf.grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize(
    "pooling_kwargs",
    [{"pool_factor": 2}, {"max_tokens": 4}, {"distance_threshold": 1.1}],
    ids=["factor", "budget", "distance"],
)
def test_single_pooling_mode_matches_pooled_loss(build_inner, pooling_kwargs) -> None:
    pooling = HierarchicalTokenPooling(**pooling_kwargs)
    wrapped_features = _build_features()
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, build_inner(model), token_poolings=[pooling])
    value = loss(wrapped_features, None)
    value.backward()

    manual_features = _build_features()
    manual_value = _mnrl(_PassthroughModel())(_pool_reference(manual_features, pooling), None)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for wrapped_leaf, manual_leaf in zip(_leaves(wrapped_features), _leaves(manual_features)):
        assert wrapped_leaf.grad.abs().sum() > 0
        assert torch.allclose(wrapped_leaf.grad, manual_leaf.grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_mixed_pooling_modes_match_manual_gradients(build_inner) -> None:
    token_poolings = [
        None,
        HierarchicalTokenPooling(pool_factor=2),
        HierarchicalTokenPooling(max_tokens=4),
        HierarchicalTokenPooling(distance_threshold=1.1),
    ]
    weights = [0.5, 1 / 6, 1 / 6, 1 / 6]
    wrapped_features = _build_features()
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, build_inner(model), token_poolings=token_poolings, weights=weights
    )
    value = loss(wrapped_features, None)
    value.backward()

    manual_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    manual_value = sum(
        weight * plain(_pool_reference(manual_features, pooling), None)
        for weight, pooling in zip(weights, token_poolings)
    )
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for wrapped_leaf, manual_leaf in zip(_leaves(wrapped_features), _leaves(manual_features)):
        assert wrapped_leaf.grad.abs().sum() > 0
        assert torch.allclose(wrapped_leaf.grad, manual_leaf.grad, atol=1e-6)


def test_token_pooling_with_kd_loss() -> None:
    """Teacher scores are per document column and unaffected by pooling, so the same labels apply
    to every variant. This is the LateOn-style KD-under-pooling recipe."""
    labels = torch.tensor([[5.0, 1.0], [4.0, 0.5], [3.0, 0.2], [6.0, 2.0]])
    pooling = HierarchicalTokenPooling(pool_factor=2)

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, kd, token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    value = loss(wrapped_features, labels)
    value.backward()

    manual_features = _build_features()
    plain = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    manual_value = plain(manual_features, labels)
    manual_value = manual_value + plain(_pool_reference(manual_features, pooling), labels)
    manual_value.backward()

    assert torch.allclose(value, manual_value, atol=1e-6)
    for leaf, manual_sf in zip(leaves, manual_features):
        assert torch.allclose(leaf.grad, manual_sf["token_embeddings"].grad, atol=1e-6)


def test_stacked_modifiers_cached_matches_non_cached() -> None:
    """Both loss modifiers stack around one loss: the Matryoshka wrapper truncates the dimension
    axis and the token pooling wrapper compresses the token axis. The stacked gradient-cached path
    must match the stacked non-cached path, which pins that both decorations land on the innermost
    cached loss in the right order."""

    def build_stacked(model: nn.Module, inner: nn.Module) -> nn.Module:
        truncated = mve_losses.MultiVectorMatryoshkaLoss(model, inner, matryoshka_dims=[DIM, DIM // 2])
        return mve_losses.MultiVectorTokenPoolingLoss(
            model,
            truncated,
            token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)],
        )

    plain_features = _build_features()
    plain_leaves = _leaves(plain_features)
    plain_model = _PassthroughModel()
    plain_value = build_stacked(plain_model, _mnrl(plain_model))(plain_features, None)
    plain_value.backward()

    cached_features = _build_features()
    cached_leaves = _leaves(cached_features)
    cached_model = _PassthroughModel()
    cached_value = build_stacked(cached_model, _cached_mnrl(cached_model))(cached_features, None)
    cached_value.backward()

    assert torch.isfinite(plain_value)
    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_leaf, cached_leaf in zip(plain_leaves, cached_leaves):
        assert plain_leaf.grad.abs().sum() > 0
        assert torch.allclose(plain_leaf.grad, cached_leaf.grad, atol=1e-6)


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_matryoshka_outside_token_pooling_raises(build_inner) -> None:
    model = _PassthroughModel()
    pooled = mve_losses.MultiVectorTokenPoolingLoss(
        model, build_inner(model), token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    with pytest.raises(ValueError, match="MultiVectorTokenPoolingLoss must wrap MultiVectorMatryoshkaLoss"):
        mve_losses.MultiVectorMatryoshkaLoss(model, pooled, matryoshka_dims=[DIM, DIM // 2])


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
def test_stacked_modifiers_match_encoding(mve_bert_tiny_model, build_inner) -> None:
    from sentence_transformers.util import batch_to_device

    model = mve_bert_tiny_model
    model.eval()
    model.prompts = {}
    model.default_prompt_name = None
    queries = ["what is late interaction", "who wrote hamlet"]
    documents = [
        "late interaction compares each query token with the most similar document token",
        "william shakespeare wrote hamlet and many other plays for the london theatre",
    ]
    dims = [model.get_embedding_dimension(), model.get_embedding_dimension() // 2]
    dim_weights = [0.4, 0.6]
    poolings = [None, HierarchicalTokenPooling(max_tokens=4)]
    pooling_weights = [0.3, 0.7]
    truncated = mve_losses.MultiVectorMatryoshkaLoss(
        model, build_inner(model, scale=1.0), matryoshka_dims=dims, matryoshka_weights=dim_weights
    )
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, truncated, token_poolings=poolings, weights=pooling_weights)
    features = [
        batch_to_device(model.preprocess(texts, task=task), model.device)
        for texts, task in [(queries, "query"), (documents, "document")]
    ]

    with torch.no_grad():
        actual = loss(features, None)
        expected = 0.0
        for pooling_weight, pooling in zip(pooling_weights, poolings):
            for dim_weight, dim in zip(dim_weights, dims):
                query_embeddings = model.encode_query(queries, token_pooling=pooling, truncate_dim=dim)
                document_embeddings = model.encode_document(documents, token_pooling=pooling, truncate_dim=dim)
                scores = model.similarity(query_embeddings, document_embeddings)
                labels = torch.arange(len(queries), device=model.device)
                expected += pooling_weight * dim_weight * F.cross_entropy(scores, labels)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_empty_variant_list_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="at least one"):
        mve_losses.MultiVectorTokenPoolingLoss(model, _mnrl(model), token_poolings=[])


def test_non_pooling_entry_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="BaseTokenPooling"):
        mve_losses.MultiVectorTokenPoolingLoss(model, _mnrl(model), token_poolings=[nn.Identity()])


@pytest.mark.parametrize("build_inner", [_mnrl, _cached_mnrl], ids=["mnrl", "cached_mnrl"])
@pytest.mark.parametrize("trainable", [False, True])
def test_pooler_parameters_and_outer_gradient_scaling(build_inner, trainable: bool) -> None:
    class LearnedPooler(BaseTokenPooling):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.7, 1.3, DIM), requires_grad=trainable)

        def pool_one(self, embedding: Tensor) -> Tensor:
            return embedding[::2] * self.weight

    model = _PassthroughModel()
    inner = build_inner(model)
    pooling = LearnedPooler()
    if build_inner is _cached_mnrl and trainable:
        calculate_loss = inner.calculate_loss
        with pytest.raises(ValueError, match="token_poolings with trainable parameters"):
            mve_losses.MultiVectorTokenPoolingLoss(model, inner, token_poolings=[None, pooling])
        assert inner.calculate_loss == calculate_loss
        return

    features = _build_features()
    loss = mve_losses.MultiVectorTokenPoolingLoss(model, inner, token_poolings=[None, pooling], weights=[0.3, 0.7])
    value = loss(features, None)
    (value / 4).backward()

    reference_pooling = LearnedPooler()
    reference_features = _build_features()
    plain = _mnrl(_PassthroughModel())
    expected = 0.3 * plain(reference_features, None)
    expected = expected + 0.7 * plain(_pool_reference(reference_features, reference_pooling), None)
    (expected / 4).backward()

    torch.testing.assert_close(value, expected)
    for actual, reference in zip(_leaves(features), _leaves(reference_features)):
        torch.testing.assert_close(actual.grad, reference.grad)
    if trainable:
        assert pooling.weight.grad.abs().sum() > 0
        torch.testing.assert_close(pooling.weight.grad, reference_pooling.weight.grad)
    else:
        assert pooling.weight.grad is None


def test_weights_length_mismatch_raises() -> None:
    model = _PassthroughModel()
    with pytest.raises(ValueError, match="one weight per pooling variant"):
        mve_losses.MultiVectorTokenPoolingLoss(
            model,
            _mnrl(model),
            token_poolings=[HierarchicalTokenPooling(pool_factor=1), HierarchicalTokenPooling(pool_factor=2)],
            weights=[1.0],
        )


def test_get_config_dict_describes_variants() -> None:
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model,
        _mnrl(model),
        token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)],
        weights=[1.0, 0.5],
    )
    config = loss.get_config_dict()
    assert config["loss"] == "MultiVectorMultipleNegativesRankingLoss"
    assert config["weights"] == (1.0, 0.5)
    assert config["n_poolings_per_step"] == -1
    assert config["token_poolings"] == [
        None,
        "HierarchicalTokenPooling(pool_factor=2, max_tokens=None, distance_threshold=None, num_protected_tokens=1, normalize_embeddings=True, "
        "tasks=['document'])",
    ]


def test_lambda_pooling_composes() -> None:
    """Any BaseTokenPooling composes, e.g. a torch-only LambdaTokenPooling, and its pooled terms
    carry gradient back to the input embeddings."""

    def halve(emb: Tensor) -> Tensor:
        n = emb.size(0)
        return emb[: n - n % 2].view(n // 2, 2, -1).mean(dim=1)

    wrapped_features = _build_features()
    leaves = _leaves(wrapped_features)
    model = _PassthroughModel()
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model,
        _mnrl(model),
        token_poolings=[None, LambdaTokenPooling(pool_fn=halve)],
    )
    value = loss(wrapped_features, None)
    assert torch.isfinite(value)
    value.backward()
    for leaf in leaves:
        assert leaf.grad.abs().sum() > 0


def test_token_pooling_on_a_real_model() -> None:
    """End to end on a real MultiVectorEncoder pipeline: the wrapper patches ``model.forward``, the
    variants pool on top of the model's scoring mask, and gradients must reach the transformer."""
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.util import batch_to_device

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.train()
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
    loss = mve_losses.MultiVectorTokenPoolingLoss(
        model, inner, token_poolings=[None, HierarchicalTokenPooling(pool_factor=2)]
    )
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    grads = [p.grad for p in model[0].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert model.forward.__self__ is model, "the original model.forward must be restored"
