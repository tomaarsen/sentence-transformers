from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import CommitInfo, HfApi, RepoUrl
from pytest import FixtureRequest

from sentence_transformers import CrossEncoder
from tests.utils import SafeTemporaryDirectory


def test_classifier_dropout_is_set() -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", classifier_dropout=0.1234)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.config.classifier_dropout == 0.1234


def test_classifier_dropout_default_value() -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None


def test_load_with_revision() -> None:
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    main_model = CrossEncoder(model_name, num_labels=1, revision="main")
    latest_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="f3cb857cba53019a20df283396bcca179cf051a4",
    )
    older_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24",
    )

    # Set the classifier.bias and classifier.weight equal among models. This
    # is needed because the AutoModelForSequenceClassification randomly initializes
    # the classifier.bias and classifier.weight for each (model) initialization.
    # The test is only possible if all models have the same classifier.bias
    # and classifier.weight parameters.
    latest_model.model.classifier.bias = main_model.model.classifier.bias
    latest_model.model.classifier.weight = main_model.model.classifier.weight
    older_model.model.classifier.bias = main_model.model.classifier.bias
    older_model.model.classifier.weight = main_model.model.classifier.weight

    test_sentences = [["Hello there!", "Hello, World!"]]
    main_prob = main_model.predict(test_sentences, convert_to_tensor=True)
    assert torch.equal(main_prob, latest_model.predict(test_sentences, convert_to_tensor=True))
    assert not torch.equal(main_prob, older_model.predict(test_sentences, convert_to_tensor=True))


@pytest.mark.parametrize(
    argnames="return_documents",
    argvalues=[True, False],
    ids=["return-docs", "no-return-docs"],
)
def test_rank(return_documents: bool, request: FixtureRequest) -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    expected_ranking = [0, 1, 3, 6, 2, 5, 7, 4, 8]

    # 1. We rank all sentences in the corpus for the query
    ranks = model.rank(query=query, documents=corpus, return_documents=return_documents)
    if request.node.callspec.id == "return-docs":
        assert {*corpus} == {rank.get("text") for rank in ranks}

    pred_ranking = [rank["corpus_id"] for rank in ranks]
    assert pred_ranking == expected_ranking


def test_rank_multiple_labels():
    model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "CrossEncoder.rank() only works for models with num_labels=1. "
            "Consider using CrossEncoder.predict() with input pairs instead."
        ),
    ):
        model.rank(
            query="A man is eating pasta.",
            documents=[
                "A man is eating food.",
                "A man is eating a piece of bread.",
                "The girl is carrying a baby.",
            ],
        )


def test_predict_softmax():
    model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
    ]
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=True, convert_to_tensor=True)
    assert torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()
    scores = model.predict([(query, doc) for doc in corpus], apply_softmax=False, convert_to_tensor=True)
    assert not torch.isclose(scores.sum(1), torch.ones(len(corpus), device=scores.device)).all()


@pytest.mark.parametrize(
    "model_name", ["cross-encoder-testing/reranker-bert-tiny-gooaq-bce", "cross-encoder/nli-MiniLM2-L6-H768"]
)
def test_predict_single_input(model_name: str):
    model = CrossEncoder(model_name)
    nested_pair_score = model.predict([["A man is eating pasta.", "A man is eating food."]])
    assert isinstance(nested_pair_score, np.ndarray)
    if model.num_labels == 1:
        assert nested_pair_score.shape == (1,)
    else:
        assert nested_pair_score.shape == (1, model.num_labels)

    pair_score = model.predict(["A man is eating pasta.", "A man is eating food."])
    if model.num_labels == 1:
        assert isinstance(pair_score, np.float32)
    else:
        assert isinstance(pair_score, np.ndarray)
        assert pair_score.shape == (model.num_labels,)


@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_numpy", [True, False])
def test_predict_output_types(
    convert_to_tensor: bool,
    convert_to_numpy: bool,
) -> None:
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
    embeddings = model.predict(
        [["One sentence", "Another sentence"]],
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )
    if convert_to_tensor:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, torch.Tensor)
    elif convert_to_numpy:
        assert embeddings[0].dtype == np.float32
        assert isinstance(embeddings, np.ndarray)
    else:
        assert embeddings[0].dtype == torch.float32
        assert isinstance(embeddings, list)


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(safe_serialization: bool) -> None:
    with SafeTemporaryDirectory() as cache_folder:
        model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
        if safe_serialization:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save_pretrained(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            model.save_pretrained(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            assert 1 == len(model_files)


def test_bfloat16() -> None:
    model = CrossEncoder(
        "cross-encoder-testing/reranker-bert-tiny-gooaq-bce", automodel_args={"torch_dtype": torch.bfloat16}
    )
    score = model.predict([["Hello there!", "Hello, World!"]])
    assert isinstance(score, np.ndarray)

    ranking = model.rank("Hello there!", ["Hello, World!", "Heya!"])
    assert isinstance(ranking, list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_assignment(device):
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device=device)
    assert model.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_device_switching():
    # test assignment using .to
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"
    assert model.model.device.type == "cpu"

    model.to("cuda")
    assert model.device.type == "cuda"
    assert model.model.device.type == "cuda"

    del model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_target_device_backwards_compat():
    model = CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device="cpu")
    assert model.device.type == "cpu"

    assert model._target_device.type == "cpu"
    model._target_device = "cuda"
    assert model.device.type == "cuda"


def test_num_labels_fresh_model():
    model = CrossEncoder("prajjwal1/bert-tiny")
    assert model.num_labels == 1


def test_push_to_hub(
    reranker_bert_tiny_model: CrossEncoder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    model = reranker_bert_tiny_model

    def mock_create_repo(self, repo_id, **kwargs):
        return RepoUrl(f"https://huggingface.co/{repo_id}")

    mock_upload_folder_kwargs = {}

    def mock_upload_folder(self, **kwargs):
        nonlocal mock_upload_folder_kwargs
        mock_upload_folder_kwargs = kwargs
        if kwargs.get("revision") is None:
            revision = "123456"
        else:
            revision = "678901"
        return CommitInfo(
            commit_url=f"https://huggingface.co/{kwargs.get('repo_id')}/commit/{revision}",
            commit_message="commit_message",
            commit_description="commit_description",
            oid="oid",
            pr_url=f"https://huggingface.co/{kwargs.get('repo_id')}/discussions/123",
        )

    def mock_create_branch(self, repo_id, branch, revision=None, **kwargs):
        return None

    monkeypatch.setattr(HfApi, "create_repo", mock_create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", mock_upload_folder)
    monkeypatch.setattr(HfApi, "create_branch", mock_create_branch)

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", revision="revision_test")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert mock_upload_folder_kwargs["revision"] == "revision_test"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/678901"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", create_pr=True)
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/discussions/123"
    mock_upload_folder_kwargs.clear()

    url = model.push_to_hub("cross-encoder-testing/stsb-distilroberta-base", tags="test-push-to-hub-tag-1")
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()
    assert "test-push-to-hub-tag-1" in model.model_card_data.tags

    url = model.push_to_hub(
        "cross-encoder-testing/stsb-distilroberta-base", tags=["test-push-to-hub-tag-2", "test-push-to-hub-tag-3"]
    )
    assert mock_upload_folder_kwargs["repo_id"] == "cross-encoder-testing/stsb-distilroberta-base"
    assert url == "https://huggingface.co/cross-encoder-testing/stsb-distilroberta-base/commit/123456"
    mock_upload_folder_kwargs.clear()
    assert "test-push-to-hub-tag-2" in model.model_card_data.tags
    assert "test-push-to-hub-tag-3" in model.model_card_data.tags
