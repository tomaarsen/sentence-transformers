import functools
import time
import requests
from torch import Tensor, device
from typing import List, Callable, Literal, Tuple, TYPE_CHECKING
from tqdm.autonotebook import tqdm
import sys
import importlib
import os
import torch
import numpy as np
import queue
import logging
from typing import Dict, Optional, Union

from transformers import is_torch_npu_available
from huggingface_hub import snapshot_download, hf_hub_download
import heapq

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import faiss
    import usearch


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i])

    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise cossim cos_sim(a[i], b[i])

    :return: Vector with res[i] = cos_sim(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance;
    see AnglELoss or https://arxiv.org/abs/2309.12871v1
    for more information.

    :return: Vector with res[i] = angle_sim(a[i], b[i])
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def paraphrase_mining(
    model, sentences: List[str], show_progress_bar: bool = False, batch_size: int = 32, *args, **kwargs
) -> List[List[Union[float, int]]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param model: SentenceTransformer model for embedding computation
    :param sentences: A list of strings (texts or sentences)
    :param show_progress_bar: Plotting of a progress bar
    :param batch_size: Number of texts that are encoded simultaneously by the model
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences
    embeddings = model.encode(
        sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_tensor=True
    )

    return paraphrase_mining_embeddings(embeddings, *args, **kwargs)


def paraphrase_mining_embeddings(
    embeddings: Tensor,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> List[List[Union[float, int]]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings: A tensor with the embeddings
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(
                embeddings[query_start_idx : query_start_idx + query_chunk_size],
                embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, sorted_i, sorted_j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs) -> List[List[Dict[str, Union[int, float]]]]:
    """This function is deprecated. Use semantic_search instead"""
    return semantic_search(*args, **kwargs)


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> List[List[Dict[str, Union[int, float]]]]:
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list


def semantic_search_faiss(
    query_embeddings: np.ndarray,
    corpus_embeddings: Optional[np.ndarray] = None,
    corpus_index: Optional["faiss.Index"] = None,
    corpus_precision: Literal["float32", "uint8", "ubinary"] = "float32",
    top_k: int = 10,
    ranges: Optional[np.ndarray] = None,
    calibration_embeddings: Optional[np.ndarray] = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> Tuple[List[List[Dict[str, Union[int, float]]]], float, "faiss.Index"]:
    """
    Performs semantic search using the FAISS library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32

    :param query_embeddings: Embeddings of the query sentences. Ideally not quantized to allow for rescoring.
    :type query_embeddings: np.ndarray
    :param corpus_embeddings: Embeddings of the corpus sentences. Either `corpus_embeddings` or `corpus_index` should
        be used, not both. The embeddings can be quantized to "int8" or "binary" for more efficient search.
    :type corpus_embeddings: Optional[np.ndarray]
    :param corpus_index: FAISS index for the corpus sentences. Either `corpus_embeddings` or `corpus_index` should
        be used, not both.
    :type corpus_index: Optional["faiss.Index"]
    :param corpus_precision: Precision of the corpus embeddings. The options are "float32", "int8", or "binary".
        Default is "float32".
    :type corpus_precision: Literal["float32", "uint8", "ubinary"]
    :param top_k: Number of top results to retrieve. Default is 10.
    :type top_k: int
    :param ranges: Ranges for quantization of embeddings. This is only used for int8 quantization, where the ranges
        refers to the minimum and maximum values for each dimension. So, it's a 2D array with shape (2, embedding_dim).
        Default is None, which means that the ranges will be calculated from the calibration embeddings.
    :type ranges: Optional[np.ndarray]
    :param calibration_embeddings: Embeddings used for calibration during quantization. This is only used for int8
        quantization, where the calibration embeddings can be used to compute ranges, i.e. the minimum and maximum
        values for each dimension. Default is None, which means that the ranges will be calculated from the query
        embeddings. This is not recommended.
    :type calibration_embeddings: Optional[np.ndarray]
    :param rescore: Whether to perform rescoring. Note that rescoring still will only be used if the query embeddings
        are not quantized and the corpus is quantized, i.e. the corpus precision is not "float32". Default is True.
    :type rescore: bool
    :param rescore_multiplier: Oversampling factor for rescoring. The code will now search `top_k * rescore_multiplier` samples
        and then rescore to only keep `top_k`. Default is 2.
    :type rescore_multiplier: int
    :param exact: Whether to use exact search or approximate search. Default is True.
    :type exact: bool
    :param output_index: Whether to output the FAISS index used for the search. Default is False.
    :type output_index: bool

    :return: A tuple containing a list of search results and the time taken for the search. If `output_index` is True,
        the tuple will also contain the FAISS index used for the search.
    :rtype: Tuple[List[List[Dict[str, Union[int, float]]]], float] or Tuple[List[List[Dict[str, Union[int, float]]]], float, "faiss.Index"]
    :raises ValueError: If both `corpus_embeddings` and `corpus_index` are provided or if neither is provided.

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    import faiss

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision == "float32":
            if exact:
                corpus_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            else:
                corpus_index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16)

        elif corpus_precision == "uint8":
            if exact:
                corpus_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            else:
                corpus_index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16)

        elif corpus_precision == "ubinary":
            if exact:
                corpus_index = faiss.IndexBinaryFlat(corpus_embeddings.shape[1] * 8)
            else:
                corpus_index = faiss.IndexBinaryHNSW(corpus_embeddings.shape[1] * 8, 16)

        corpus_index.add(corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # oversampling factor
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if corpus_precision != "float32" and rescore:
            rescore_embeddings = query_embeddings
            k *= rescore_multiplier

        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )

    # Perform the search using the usearch index
    start_t = time.time()
    scores, indices = corpus_index.search(query_embeddings, k)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array(
            [[corpus_index.reconstruct(idx.item()) for idx in query_indices] for query_indices in indices]
        )
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision == "ubinary":
            top_k_embeddings = np.unpackbits(top_k_embeddings, axis=-1).astype(int)
        else:
            top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[:, :top_k]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def semantic_search_usearch(
    query_embeddings: np.ndarray,
    corpus_embeddings: Optional[np.ndarray] = None,
    corpus_index: Optional["usearch.index.Index"] = None,
    corpus_precision: Literal["float32", "int8", "binary"] = "float32",
    top_k: int = 10,
    ranges: Optional[np.ndarray] = None,
    calibration_embeddings: Optional[np.ndarray] = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> Tuple[List[List[Dict[str, Union[int, float]]]], float, "usearch.index.Index"]:
    """
    Performs semantic search using the usearch library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32

    :param query_embeddings: Embeddings of the query sentences. Ideally not quantized to allow for rescoring.
    :type query_embeddings: np.ndarray
    :param corpus_embeddings: Embeddings of the corpus sentences. Either `corpus_embeddings` or `corpus_index` should
        be used, not both. The embeddings can be quantized to "int8" or "binary" for more efficient search.
    :type corpus_embeddings: Optional[np.ndarray]
    :param corpus_index: usearch index for the corpus sentences. Either `corpus_embeddings` or `corpus_index` should
        be used, not both.
    :type corpus_index: Optional["usearch.Index"]
    :param corpus_precision: Precision of the corpus embeddings. The options are "float32", "int8", or "binary".
        Default is "float32".
    :type corpus_precision: Literal["float32", "int8", "binary"]
    :param top_k: Number of top results to retrieve. Default is 10.
    :type top_k: int
    :param ranges: Ranges for quantization of embeddings. This is only used for int8 quantization, where the ranges
        refers to the minimum and maximum values for each dimension. So, it's a 2D array with shape (2, embedding_dim).
        Default is None, which means that the ranges will be calculated from the calibration embeddings.
    :type ranges: Optional[np.ndarray]
    :param calibration_embeddings: Embeddings used for calibration during quantization. This is only used for int8
        quantization, where the calibration embeddings can be used to compute ranges, i.e. the minimum and maximum
        values for each dimension. Default is None, which means that the ranges will be calculated from the query
        embeddings. This is not recommended.
    :type calibration_embeddings: Optional[np.ndarray]
    :param rescore: Whether to perform rescoring. Note that rescoring still will only be used if the query embeddings
        are not quantized and the corpus is quantized, i.e. the corpus precision is not "float32". Default is True.
    :type rescore: bool
    :param rescore_multiplier: Oversampling factor for rescoring. The code will now search `top_k * rescore_multiplier` samples
        and then rescore to only keep `top_k`. Default is 2.
    :type rescore_multiplier: int
    :param exact: Whether to use exact search or approximate search. Default is True.
    :type exact: bool
    :param output_index: Whether to output the usearch index used for the search. Default is False.
    :type output_index: bool

    :return: A tuple containing a list of search results and the time taken for the search. If `output_index` is True,
        the tuple will also contain the usearch index used for the search.
    :rtype: Tuple[List[List[Dict[str, Union[int, float]]]], float] or Tuple[List[List[Dict[str, Union[int, float]]]], float, "usearch.index.Index"]
    :raises ValueError: If both `corpus_embeddings` and `corpus_index` are provided or if neither is provided.

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    from usearch.index import Index
    from usearch.compiled import ScalarKind

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", or "binary" for usearch')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision == "float32":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="cos",
                dtype="f32",
            )
        elif corpus_precision == "int8":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="ip",
                dtype="i8",
            )
        elif corpus_precision == "binary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="hamming",
                dtype="i8",
            )
        corpus_index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # oversampling factor
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if corpus_index.dtype != ScalarKind.F32 and rescore:
            rescore_embeddings = query_embeddings
            k *= rescore_multiplier

        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )

    # Perform the search using the usearch index
    start_t = time.time()
    matches = corpus_index.search(query_embeddings, count=k, exact=exact)
    scores = matches.distances
    indices = matches.keys

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array([corpus_index.get(query_indices) for query_indices in indices])
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision == "binary":
            top_k_embeddings = np.unpackbits(top_k_embeddings.astype(np.uint8), axis=-1)
        top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[:, :top_k]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def http_get(url, path) -> None:
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def fullname(o) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files
    """

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)


def community_detection(
    embeddings, threshold=0.75, min_community_size=10, batch_size=1024, show_progress_bar=False
) -> List[List[int]]:
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


##################
#
######################


class disabled_tqdm(tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: Optional[Union[bool, str]] = None,
    cache_folder: Optional[str] = None,
    revision: Optional[str] = None,
) -> bool:
    return bool(load_file_path(model_name_or_path, "modules.json", token, cache_folder, revision=revision))


def load_file_path(
    model_name_or_path: str,
    filename: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
) -> Optional[str]:
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
        )
    except Exception:
        return


def load_dir_path(
    model_name_or_path: str,
    directory: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
) -> Optional[str]:
    # If file is local
    dir_path = os.path.join(model_name_or_path, directory)
    if os.path.exists(dir_path):
        return dir_path

    download_kwargs = {
        "repo_id": model_name_or_path,
        "revision": revision,
        "allow_patterns": f"{directory}/**",
        "library_name": "sentence-transformers",
        "token": token,
        "cache_dir": cache_folder,
        "tqdm_class": disabled_tqdm,
    }
    # Try to download from the remote
    try:
        repo_path = snapshot_download(**download_kwargs)
    except Exception:
        # Otherwise, try local (i.e. cache) only
        download_kwargs["local_files_only"] = True
        repo_path = snapshot_download(**download_kwargs)
    return os.path.join(repo_path, directory)


def save_to_hub_args_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # If repo_id not already set, use repo_name
        repo_name = kwargs.pop("repo_name", None)
        if repo_name and "repo_id" not in kwargs:
            logger.warning(
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper


def get_device_name() -> Literal["mps", "cuda", "npu", "cpu"]:
    """
    Returns the name of the device where this module is running on.
    It's simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    :return: Device name, like 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    else:
        return "cpu"


def quantize_embeddings(
    embeddings: Union[Tensor, np.ndarray],
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: Optional[np.ndarray] = None,
    calibration_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    :param embeddings: Unquantized (e.g. float) embeddings with to quantize to a given precision
    :param precision: The precision to convert to. Options are "float32", "int8", "uint8", "binary", "ubinary".
    :param ranges: Ranges for quantization of embeddings. This is only used for int8 quantization, where the ranges
        refers to the minimum and maximum values for each dimension. So, it's a 2D array with shape (2, embedding_dim).
        Default is None, which means that the ranges will be calculated from the calibration embeddings.
    :type ranges: Optional[np.ndarray]
    :param calibration_embeddings: Embeddings used for calibration during quantization. This is only used for int8
        quantization, where the calibration embeddings can be used to compute ranges, i.e. the minimum and maximum
        values for each dimension. Default is None, which means that the ranges will be calculated from the query
        embeddings. This is not recommended.
    :type calibration_embeddings: Optional[np.ndarray]
    :return: Quantized embeddings with the specified precision
    """
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255

        if precision == "uint8":
            return ((embeddings - starts) / steps).astype(np.uint8)
        elif precision == "int8":
            return ((embeddings - starts) / steps - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)

    raise ValueError(f"Precision {precision} is not supported")
