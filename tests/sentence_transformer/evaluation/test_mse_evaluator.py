from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sentence_transformers.sentence_transformer.evaluation import MSEEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize(("source_count", "target_count"), [(1, 2), (2, 1), (2, 3), (0, 1), (1, 0)])
def test_mse_evaluator_rejects_unequal_sentence_counts(evaluator_cls, source_count, target_count):
    teacher = Mock()
    teacher.sparsity.return_value = {}

    with pytest.raises(ValueError, match="source_sentences and target_sentences must have the same length"):
        evaluator_cls(["source"] * source_count, ["target"] * target_count, teacher_model=teacher)

    teacher.encode.assert_not_called()


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize(("sentence_count", "expected_score"), [(1, -100.0), (2, -50.0)])
def test_mse_evaluator_equal_sentence_counts(evaluator_cls, sentence_count, expected_score):
    teacher, student = Mock(), Mock()
    as_embeddings = torch.tensor if evaluator_cls is SparseMSEEvaluator else np.array
    teacher.encode.return_value = as_embeddings([[0.0, 1.0], [1.0, 0.0]])[:sentence_count]
    student.encode.return_value = as_embeddings([[1.0, 0.0], [1.0, 0.0]])[:sentence_count]
    teacher.sparsity.return_value = student.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}

    evaluator = evaluator_cls(["source"] * sentence_count, ["target"] * sentence_count, teacher_model=teacher)

    assert evaluator(student)["negative_mse"] == pytest.approx(expected_score)
