from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class ListMLELambdaWeight(nn.Module):
    """Base class for implementing weighting schemes in Position-Aware ListMLE Loss."""

    def __init__(self, rank_discount_fn=None) -> None:
        """
        Initialize a lambda weight for ListMLE loss.

        Args:
            rank_discount_fn: Function that computes a discount for each rank position.
                              If None, uses default discount of 2^(list_size - rank) - 1.
        """
        super().__init__()
        self.rank_discount_fn = rank_discount_fn

    def forward(self, ranks: Tensor, list_size: int) -> Tensor:
        """
        Calculate position-aware weights for the ListMLE loss.

        Args:
            ranks: A tensor of rank positions [batch_size, list_size]
            list_size: Size of the list

        Returns:
            Tensor: Weights for each position [batch_size, list_size]
        """
        if self.rank_discount_fn is not None:
            return self.rank_discount_fn(ranks)

        # Default rank discount: 2^(list_size - rank) - 1
        return torch.pow(2.0, list_size - ranks) - 1.0


class ListMLELoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        lambda_weight: ListMLELambdaWeight | None = None,
        activation_fct: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
        respect_input_order: bool = True,
    ) -> None:
        """
        ListMLE loss for learning to rank with position-aware weighting. This loss function implements 
        the ListMLE ranking algorithm which uses a list-wise approach based on maximum likelihood 
        estimation of permutations. It maximizes the likelihood of the permutation induced by the 
        ground truth labels with optional position-aware weighting.

        .. note::

            The number of documents per query can vary between samples with the ``ListMLELoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            lambda_weight (ListMLELambdaWeight, optional): Weighting scheme to use. When specified,
                implements Position-Aware ListMLE which applies different weights to different rank 
                positions. Default is None (standard ListMLE).
            activation_fct (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:

                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.

                Defaults to None.
            respect_input_order (bool): Whether to respect the original input order of documents.
                If True, assumes the input documents are already ordered by relevance (most relevant first).
                If False, sorts documents by label values. Defaults to True.

        References:
            - Listwise approach to learning to rank: theory and algorithm: https://dl.acm.org/doi/abs/10.1145/1390156.1390306
            - Position-Aware ListMLE: A Sequential Learning Process for Ranking: https://auai.org/uai2014/proceedings/individuals/164.pdf
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Texts                                  | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "docs": [
                        ["Pandas are a kind of bear.", "Pandas are kind of like fish."],
                        ["The capital of France is Paris.", "Paris is the capital of France.", "Paris is quite large."],
                    ],
                    "labels": [[1, 0], [1, 1, 0]],
                })
                
                # Standard ListMLE loss respecting input order
                loss = losses.ListMLELoss(model)
                
                # Position-Aware ListMLE with default weighting
                lambda_weight = losses.ListMLELambdaWeight()
                loss = losses.ListMLELoss(model, lambda_weight=lambda_weight)
                
                # Position-Aware ListMLE with custom weighting function
                def custom_discount(ranks): # e.g. ranks: [1, 2, 3, 4, 5]
                    return 1.0 / torch.log1p(ranks)
                lambda_weight = losses.ListMLELambdaWeight(rank_discount_fn=custom_discount)
                loss = losses.ListMLELoss(model, lambda_weight=lambda_weight)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.lambda_weight = lambda_weight
        self.activation_fct = activation_fct or nn.Identity()
        self.mini_batch_size = mini_batch_size
        self.respect_input_order = respect_input_order
        self.eps = 1e-10

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute ListMLE loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Mean ListMLE loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "ListMLELoss expects a list of labels for each sample, but got a single value for each sample."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"ListMLELoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
            )

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            # Handle edge case where there are no documents
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
        logits = self.activation_fct(logits)

        # Create output tensor filled with a very small value for padded logits
        logits_matrix = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        # Create a mask for valid entries
        mask = torch.zeros_like(logits_matrix, dtype=torch.bool)
        mask[batch_indices, doc_indices] = True

        # Convert labels to tensor matrix
        labels_matrix = torch.full_like(logits_matrix, -float("inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()
        
        if not torch.any(mask):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        if not self.respect_input_order:
            # Sort by labels in descending order if not respecting input order.
            sorted_labels, indices = labels_matrix.sort(descending=True, dim=1)
            sorted_logits = torch.gather(logits_matrix, 1, indices)
        else:
            # Use the original input order, assuming it's already ordered by relevance
            sorted_logits = logits_matrix

        # Compute log-likelihood using Plackett-Luce model
        scores = sorted_logits.exp()
        cumsum_scores = torch.flip(torch.cumsum(torch.flip(scores, [1]), 1), [1])
        log_probs = sorted_logits - torch.log(cumsum_scores + self.eps)

        # Apply position-aware lambda weights if specified
        if self.lambda_weight is not None:
            position_weights = torch.zeros_like(log_probs)
            for i, query_mask in enumerate(mask):
                list_size = query_mask.sum()
                ranks = torch.arange(1, list_size + 1, device=self.model.device)
                position_weights[i, :list_size] = self.lambda_weight(ranks, list_size)
            log_probs = log_probs * position_weights

        # Sum the log probabilities for each list and mask invalid entries
        per_query_losses = -torch.sum(log_probs, dim=1)

        if not torch.any(per_query_losses):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)
            
        # Average loss over all lists
        return torch.mean(per_query_losses)

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "lambda_weight": None if self.lambda_weight is None else fullname(self.lambda_weight),
            "activation_fct": fullname(self.activation_fct),
            "mini_batch_size": self.mini_batch_size,
            "respect_input_order": self.respect_input_order,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{xia2008listwise,
  title={Listwise approach to learning to rank: theory and algorithm},
  author={Xia, Fen and Liu, Tie-Yan and Wang, Jue and Zhang, Wensheng and Li, Hang},
  booktitle={Proceedings of the 25th international conference on Machine learning},
  pages={1192--1199},
  year={2008}
},
@inproceedings{lan2014position,
  title={Position-Aware ListMLE: A Sequential Learning Process for Ranking.},
  author={Lan, Yanyan and Zhu, Yadong and Guo, Jiafeng and Niu, Shuzi and Cheng, Xueqi},
  booktitle={UAI},
  volume={14},
  pages={449--458},
  year={2014}
}
"""