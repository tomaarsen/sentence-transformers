from __future__ import annotations

import logging
import os
from ast import List
from typing import TYPE_CHECKING, Literal

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

    try:
        from mteb.abstasks.TaskMetadata import TASK_CATEGORY, TASK_DOMAIN, TASK_TYPE
        from mteb.benchmarks import BENCHMARK_REGISTRY
        from mteb.custom_validators import MODALITIES

        BENCHMARKS = Literal[*BENCHMARK_REGISTRY.keys()]
    except ImportError:
        TASK_DOMAIN = str
        TASK_TYPE = str
        TASK_CATEGORY = str
        MODALITIES = str
        BENCHMARKS = str

logger = logging.getLogger(__name__)


class MTEBEvaluator(SentenceEvaluator):
    """

    Args:
        benchmark: The MTEB benchmark to evaluate on.
        tasks: A list of task names to include. If None, all tasks which pass the filters are included.
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn". For multilingual tasks this will also remove languages that are not in the specified list.
        script: A list of script codes (ISO 15924 codes). If None, all scripts are included. For multilingual tasks this will also remove scripts
            that are not in the specified list.
        domains: A list of task domains.
        task_types: A string specifying the type of task. If None, all tasks are included.
        categories: A list of task categories these include "s2s" (sentence to sentence), "s2p" (sentence to paragraph) and "p2p" (paragraph to
            paragraph).
        exclude_superseded: A boolean flag to exclude datasets which are superseded by another.
        eval_splits: A list of evaluation splits to include. If None, all splits are included.
        exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
            exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
            specified will be kept.
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.
        exclude_aggregate: If True, exclude aggregate tasks. If False, both aggregate and non-aggregate tasks are returned.

        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 32.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation
            dimension. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import MTEBEvaluator

    """

    def __init__(
        self,
        benchmark: List[BENCHMARKS] = None,
        tasks: list[str] | None = None,
        languages: list[str] | None = None,
        script: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,  # type: ignore
        task_types: list[TASK_TYPE] | None = None,  # type: ignore
        categories: list[TASK_CATEGORY] | None = None,  # type: ignore
        exclude_superseded: bool = True,
        eval_splits: list[str] | None = None,
        exclusive_language_filter: bool = False,
        modalities: list[MODALITIES] | None = None,  # type: ignore
        exclusive_modality_filter: bool = False,
        exclude_aggregate: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        truncate_dim: int | None = None,
        **mteb_kwargs,
    ):
        super().__init__()
        try:
            import mteb
        except ImportError:
            raise ImportError(
                "MTEB is not installed. Please install it with `pip install mteb` or `pip install -U sentence-transformers` "
                "to use the MTEBEvaluator."
            )

        self.benchmark = benchmark
        if self.benchmark is not None:
            mteb_tasks = mteb.get_benchmark(self.benchmark).tasks
        else:
            mteb_tasks = mteb.get_tasks(
                tasks=tasks,
                languages=languages,
                script=script,
                domains=domains,
                task_types=task_types,
                categories=categories,
                exclude_superseded=exclude_superseded,
                eval_splits=eval_splits,
                exclusive_language_filter=exclusive_language_filter,
                modalities=modalities,
                exclusive_modality_filter=exclusive_modality_filter,
                exclude_aggregate=exclude_aggregate,
                **mteb_kwargs,
            )
        self.evaluation = mteb.MTEB(tasks=mteb_tasks)

        self.truncate_dim = truncate_dim
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.primary_metric = "MTEB_average"

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch=-1, steps=-1) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        output_path = output_path or "results"
        if epoch != -1 and steps != -1:
            output_path = os.path.join(output_path, f"epoch_{epoch}_steps_{steps}")

        # To avoid potential errors, we inject a mteb_model_meta in the MTEBEvaluator on the Sentence Transformer side,
        # as the MTEB library may have issues with e.g. the format of the languages or the model name.
        self.set_mteb_model_meta(model)

        results = self.evaluation.run(
            model,
            output_path=output_path,
            encode_kwargs={"truncate_dim": self.truncate_dim, "batch_size": self.batch_size},
            verbosity=1 if self.show_progress_bar else 0,
        )
        metrics = {}
        for task, task_result in zip(self.evaluation.tasks, results):
            for split, scores_list in task_result.scores.items():
                for scores in scores_list:
                    # metrics[f"{task_result.task_name}_{split}_{scores['hf_subset']}"] = scores["main_score"]
                    metrics[f"{task_result.task_name}_{task.metadata.main_score}"] = scores["main_score"]
                    # for metric, score in scores.items():
                    #     if metric == "main_score":
                    #         # TODO: primary_metric
                    #         continue
                    #     if "nauc" in metric or metric == "hf_subset" or metric == "languages":
                    #         continue
                    # metrics[f"{task_result.task_name}_{split}_{scores['hf_subset']}_{metric}"] = score

        metrics["MTEB_average"] = sum(metrics.values()) / len(metrics)
        metrics = self.prefix_name_to_metrics(metrics, None)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def set_mteb_model_meta(self, model: SentenceTransformer):
        if hasattr(model, "mteb_model_meta"):
            return

        from mteb import ModelMeta

        name = model.model_card_data.model_id if model.model_card_data.model_id else model.model_card_data.base_model
        meta = ModelMeta(
            name=name,
            revision=model.model_card_data.base_model_revision,
            release_date=None,
            languages=None,
            framework=["Sentence Transformers"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=model.max_seq_length,
            embed_dim=model.get_sentence_embedding_dimension(),
            license=model.model_card_data.license,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            similarity_fn_name=model.similarity_fn_name,
            use_instructions=None,
            training_datasets=None,
        )
        model.mteb_model_meta = meta

    @property
    def description(self) -> str:
        return "MTEB"

    @property
    def tasks(self):
        return self.evaluation.tasks

    def __repr__(self) -> str:
        return f"MTEBEvaluator(tasks={[task.metadata.name for task in self.evaluation.tasks]})"

    def __str__(self) -> str:
        task_names = [task.metadata.name for task in self.evaluation.tasks]
        return f"MTEBEvaluator with {len(task_names)} tasks: {', '.join(task_names[:3])}" + (
            "..." if len(task_names) > 3 else ""
        )

    def get_config_dict(self):
        config_dict = {}
        if self.benchmark is not None:
            config_dict["benchmark"] = self.benchmark
        else:
            config_dict["tasks"] = [task.metadata.name for task in self.evaluation.tasks]
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
