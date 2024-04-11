from . import SentenceEvaluator
from typing import Iterable


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value
    """

    def __init__(self, evaluators: Iterable[SentenceEvaluator], main_score_function=lambda scores: scores[-1]):
        super().__init__()
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        evaluations = []
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluation, "primary_metric"):
                    scores.append(evaluation[evaluation.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            evaluations.append(evaluation)

        self.primary_metric = "sequential_score"
        main_score = self.main_score_function(scores)
        results = {key: value for evaluation in evaluations for key, value in evaluation.items()}
        results["sequential_score"] = main_score
        return results
