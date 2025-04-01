from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder


class ApproxDiscountedRankMSE(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        temperature: float = 1.0,
        discount: Literal["log2", "reciprocal"] | None = None,
        activation_fn: nn.Module = nn.Identity(),
        mini_batch_size: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.discount = discount
        self.activation_fn = activation_fn or nn.Identity()
        self.mini_batch_size = mini_batch_size

        self.loss_fct = nn.MSELoss(reduction="none")

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    @staticmethod
    def get_approx_ranks(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        diff = logits[:, None] - logits[..., None]
        normalized_diff = torch.sigmoid(diff / temperature)
        # set diagonal to 0
        normalized_diff = normalized_diff * (1 - torch.eye(logits.shape[1], device=logits.device))
        approx_ranks = normalized_diff.sum(-1) + 1
        return approx_ranks

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if isinstance(labels, Tensor):
            raise ValueError(
                "LambdaLoss expects a list of labels for each sample, but got a single value for each sample."
            )
        if len(inputs) != 2:
            raise ValueError(f"LambdaLoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs.")

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        # Create input pairs for the model
        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            # Handle edge case where all documents are padded
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        mini_batch_size = self.mini_batch_size or batch_size
        if mini_batch_size <= 0:
            mini_batch_size = len(pairs)

        logits_list = []
        for i in range(0, len(pairs), mini_batch_size):
            mini_batch_pairs = pairs[i : i + mini_batch_size]

            tokens = self.model.tokenizer(
                mini_batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokens.to(self.model.device)

            logits = self.model(**tokens)[0].view(-1)
            logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)
        logits = self.activation_fn(logits)

        # Create output tensor filled with 0 (padded logits will be ignored via labels)
        logits_matrix = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        approx_ranks = self.get_approx_ranks(logits_matrix, self.temperature)
        true_ranks = torch.arange(1, approx_ranks.shape[1] + 1, device=logits.device, dtype=approx_ranks.dtype).repeat(
            batch_size, 1
        )
        loss = self.loss_fct(approx_ranks, true_ranks)
        if self.discount == "log2":
            weight = 1 / torch.log2(true_ranks + 1)
        elif self.discount == "reciprocal":
            weight = 1 / true_ranks
        else:
            weight = 1
        loss = loss * weight
        loss = loss.mean()
        return loss
