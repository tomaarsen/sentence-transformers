"""
Tests the RerankingEvaluator
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator

ROWS = [
    {"query": "What sound does a cat make?", "positive": "Cats meow.", "negative": "Dogs bark loudly at night."},
    {
        "query": "What is the capital of France?",
        "positive": "Paris is the capital of France.",
        "negative": "Berlin is in Germany.",
    },
]


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(("string_positive", "string_negative"), [(True, False), (False, True), (True, True)])
def test_reranking_evaluator_string_documents(
    stsb_bert_tiny_model: SentenceTransformer,
    use_batched_encoding: bool,
    string_positive: bool,
    string_negative: bool,
) -> None:
    """A single positive or negative document given as a string is one document, not a list of characters"""
    as_lists = [{"query": row["query"], "positive": [row["positive"]], "negative": [row["negative"]]} for row in ROWS]
    expected = RerankingEvaluator(as_lists, name="dev", use_batched_encoding=use_batched_encoding)(
        stsb_bert_tiny_model
    )

    rows = [
        {
            **row,
            "positive": row["positive"] if string_positive else [row["positive"]],
            "negative": row["negative"] if string_negative else [row["negative"]],
        }
        for row in ROWS
    ]
    for samples in (rows, Dataset.from_list(rows)):
        evaluator = RerankingEvaluator(samples, name="dev", use_batched_encoding=use_batched_encoding)
        assert evaluator(stsb_bert_tiny_model) == pytest.approx(expected)
