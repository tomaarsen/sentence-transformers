# Token Pooling-aware Training

A multi-vector retrieval model can represent a long document with hundreds of token vectors, many of which describe similar content. Token pooling merges these vectors to make the document index smaller and reduce the number of comparisons needed for MaxSim scoring. Compression can also remove distinctions that matter for retrieval, so pooling-aware training teaches the encoder to preserve useful information when its document vectors are merged.

This example uses `HierarchicalTokenPooling`, which groups similar token embeddings with Ward clustering and replaces each cluster with its normalized mean. Unlike [Matryoshka truncation](../matryoshka/README.md), which reduces the dimensionality of every vector, token pooling reduces how many vectors represent a document. Queries remain unpooled by default, preserving their individual token vectors for matching against the compressed documents. The encoder still runs its full backbone, and clustering adds work during indexing, with the savings coming from storing and scoring fewer document vectors.

## Training embeddings that pool well

Ordinary retrieval training optimizes the original token embeddings, without requiring them to remain useful after pooling. `MultiVectorTokenPoolingLoss` adds that requirement by applying a retrieval loss to the pooled representations during training. In [training_cached_contrastive.py](training_cached_contrastive.py), the same batch contributes both an unpooled retrieval loss and a loss using at most 16 vectors per document:

```python
from sentence_transformers.multi_vector_encoder.losses import (
    CachedMultiVectorMultipleNegativesRankingLoss,
    MultiVectorTokenPoolingLoss,
)
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

pooling = HierarchicalTokenPooling(max_tokens=16)
loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=64)
loss = MultiVectorTokenPoolingLoss(
    model,
    loss,
    token_poolings=[None, pooling],
    weights=[0.5, 0.5],
)
```

`CachedMultiVectorMultipleNegativesRankingLoss` supplies the retrieval objective, teaching each query to score its relevant document above the negative documents in the batch. Gradient caching lets the encoder process 64 inputs at a time while each query is compared with documents from all 256 training examples, reducing training memory without reducing the number of negatives.

`MultiVectorTokenPoolingLoss` wraps that objective and evaluates it using each entry in `token_poolings`. Here, `None` uses the original embeddings without pooling, while `pooling` compresses documents to at most 16 vectors. The weights of `0.5` simply average the two retrieval losses, teaching the encoder to support compressed retrieval alongside ordinary retrieval. The variants reuse the token embeddings without an additional encoder pass for each pooler, although clustering and computing the extra retrieval loss add some overhead.

The entries in `token_poolings` can also be custom `BaseTokenPooling` implementations, provided they preserve gradients through the input embeddings. With `HierarchicalTokenPooling`, Ward clustering makes discrete choices about which tokens to merge. During training, it uses a straight-through estimator to pass gradients back to the token embeddings without differentiating those cluster assignments. Each cluster member receives the gradient through its pooled representation, allowing the encoder to learn from errors made with compressed documents.

The example script learns retrieval and poolability together from MS MARCO triplets, starting with `answerdotai/ModernBERT-base` and a fresh 128-dimensional token projection. The same loss wrapper can also fine-tune an existing retrieval model. Setting `query_expansion={"strategy": "min", "length": 32}` provides additional contextual query vectors by expanding short queries to at least 32 token positions, while longer queries retain their length up to the training token cap. To optimize for another deployment constraint, replace the 16-vector pooler with your chosen pooler, e.g. a `HierarchicalTokenPooling` with a different `max_tokens`, `pool_factor`, or `distance_threshold` setting.

Run the training script with and without `--baseline` to compare the effect of pooling-aware training:

```bash
python -m examples.multi_vector_encoder.training.token_pooling.training_cached_contrastive --baseline
python -m examples.multi_vector_encoder.training.token_pooling.training_cached_contrastive
```

Both commands use the same initialization, data, and optimization settings, with `--baseline` changing only the loss. Small MS MARCO NanoBEIR evaluations track unpooled and 16-vector retrieval alongside held-out validation loss, helping you check whether the compressed representations are learning to retrieve. You can adapt the model, poolers, and training settings directly in the script, keeping both runs matched when comparing losses. It saves the encoder under `models/<run_name>/final` and attempts a Hub upload using your `hf auth login` credentials.

## Applying token pooling

Load the example model from the Hub or your own saved encoder, then pass the pooler when encoding documents:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("tomaarsen/mve-ModernBERT-base-token-pooling")
pooling = HierarchicalTokenPooling(max_tokens=16)

query_embeddings = model.encode_query(["How can I make a retrieval index smaller?"])
document_embeddings = model.encode_document(
    [
        "Token pooling merges similar token embeddings into fewer vectors, reducing document index storage.",
        "Paris is the capital of France.",
    ],
    token_pooling=pooling,
)
scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
```

The loss teaches the encoder to work with pooled embeddings, while `token_pooling` selects the compression applied at inference. Saving the encoder does not automatically enable the poolers used in its training loss. Apply the same pooling configuration when building the document index that you evaluated for deployment.

To use a pooler by default, pass it when initializing the encoder:

```python
model = MultiVectorEncoder("tomaarsen/mve-ModernBERT-base-token-pooling", token_pooling=pooling)
```

Calls to `encode`, `encode_query`, and `encode_document` then use this default, with the pooler's `tasks` still deciding which inputs are pooled. Pass another pooler to override it for a call, or set `model.token_pooling = None` to clear the default. To save pooling as part of the model, add the pooler to its module pipeline instead.

With `HierarchicalTokenPooling`, choose one of the three pooling controls according to the constraint you want to express:

| Control | Effect | When to use it |
| --- | --- | --- |
| `max_tokens=16` | Keep at most 16 vectors, including protected tokens.<br>Shorter documents keep their existing vectors. | You want a per-document cap. |
| `pool_factor=4` | Keep approximately one quarter<br>of each document's vectors. | You want compression proportional<br>to document length. |
| `distance_threshold` | Merge up to a shared Ward distance threshold.<br>Vector counts depend on embedding geometry. | You want adaptive compression.<br>Calibrate it for an average vector budget. |

These controls are alternatives, so configure one at a time. By default, the first token is protected from merging, and all output vectors are normalized. A cap of 16 can therefore average fewer than 16 vectors per document because shorter documents already fit within the budget.

## Measuring the quality and storage tradeoff

[evaluate_token_pooling.py](evaluate_token_pooling.py) compares unpooled retrieval with document caps of 64, 32, 16, 8, and 4 vectors on all 13 NanoBEIR datasets. Running it on any multi-vector embedding model shows how much quality is retained at the compression you intend to deploy, and you can edit the poolers and evaluation settings to suit your model and use case. For example, compare the pooling-aware model with its baseline:

```bash
python -m examples.multi_vector_encoder.training.token_pooling.evaluate_token_pooling tomaarsen/mve-ModernBERT-base-token-pooling
python -m examples.multi_vector_encoder.training.token_pooling.evaluate_token_pooling tomaarsen/mve-ModernBERT-base-token-pooling-baseline
```

The comparison below uses matched training on 120,000 MS MARCO triplets with seed 42 and the final checkpoints. Scores are mean nDCG@10 (%) across the 13 NanoBEIR datasets, with FP32 inference and storage compared at unchanged embedding dimensionality and numeric precision. Mean vectors is the average number of document vectors across the evaluation corpora.

| Pooling at inference | Mean vectors | Ordinary loss | Pooling-aware loss | Gain (points) |
| --- | ---: | ---: | ---: | ---: |
| Unpooled | 193.22 | 51.84 | 52.04 | +0.19 |
| `max_tokens=64` | 54.76 | 49.27 | 51.34 | +2.07 |
| `max_tokens=32` | 29.58 | 48.25 | 50.69 | +2.44 |
| `max_tokens=16` | 15.60 | 46.61 | 49.87 | +3.26 |
| `max_tokens=8` | 7.99 | 43.12 | 47.42 | +4.30 |
| `max_tokens=4` | 4.00 | 37.22 | 45.40 | +8.18 |
| `pool_factor=4` | 48.68 | 49.03 | 51.61 | +2.58 |
| Calibrated `distance_threshold` | 15.60 | 47.23 | 50.19 | +2.96 |

![Retrieval quality versus document vector storage](results.png)

At the trained 16-vector cap, pooling-aware training improves retrieval from 46.61 to 49.87 while reducing document vector storage by 12.4x relative to unpooled retrieval. The pooling-aware encoder at 8 vectors also outperforms the ordinary encoder at 16 vectors, illustrating how training can make a smaller index more useful. Training covered unpooled retrieval and the 16-vector cap, while the other settings show how well the learned embeddings transfer to compression settings they were not explicitly trained for. The figure and evaluation script cover the fixed caps, with the table also including proportional and adaptive pooling.

The calibrated row uses one shared `distance_threshold` per encoder and NanoBEIR corpus, chosen from that corpus's unpooled document embeddings to match the average vector count under `max_tokens=16`. Documents can receive more or fewer than 16 vectors, while using approximately the same average storage. The next section shows how to choose this threshold without relevance labels or additional training.

MS MARCO, Natural Questions, and FiQA were used for recipe development, and the full suite also informed exploratory experiments. These are therefore exploratory results on a small benchmark. A separate check with the trained models used official test queries outside NanoBEIR and each dataset's full corpus, retaining graded relevance. Both comparisons below use the 16-vector cap and the same seed-42 models:

| Additional queries | Queries | Ordinary loss | Pooling-aware loss | Gain (points) |
| --- | ---: | ---: | ---: | ---: |
| [SciFact](https://huggingface.co/datasets/BeIR/scifact) | 250 | 59.48 | 60.73 | +1.25 |
| [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | 273 | 26.12 | 26.96 | +0.83 |

## Calibrating adaptive pooling

A fixed cap like `max_tokens` allocates the same maximum number of vectors to every document, even when some documents contain many similar token embeddings and others contain more distinct information. Adaptive pooling instead uses a shared `distance_threshold` to decide when to stop merging. Documents whose embeddings can be merged at smaller Ward distances receive fewer vectors, while those requiring larger merge distances retain more. This differs from `pool_factor`, which reduces the vector count according to document length.

The storage produced by a threshold depends on the encoder and its document embeddings, so calibration translates an average vector budget into a suitable threshold. Encode a representative sample of your documents without pooling, then use `HierarchicalTokenPooling.calibrate_distance_threshold` to choose a threshold targeting an average of 16 vectors per document:

```python
calibration_embeddings = model.encode_document(calibration_documents)
threshold = HierarchicalTokenPooling.calibrate_distance_threshold(calibration_embeddings, target_tokens=16)
print(threshold)
# e.g. 0.2843...
pooling = HierarchicalTokenPooling(distance_threshold=threshold)
document_embeddings = model.encode_document(documents_to_index, token_pooling=pooling)
```

Calibration uses only document embeddings to choose a threshold you can reuse in your pooling configuration. If no compression is needed to meet the budget, it returns `None`. Individual documents can exceed the average budget, and the achieved average can change on a different corpus, so evaluate the calibrated pooler on the documents you expect to index.

<details>
<summary>Matching the average storage of a fixed cap for a fair comparison</summary>

A cap of `max_tokens=16` often averages fewer than 16 vectors because short documents already contain fewer vectors. To compare adaptive pooling at the same average storage, first measure the fixed cap's output on the calibration documents, then use that average as the calibration target:

```python
fixed_embeddings = HierarchicalTokenPooling(max_tokens=16).pool(calibration_embeddings)
target_tokens = sum(map(len, fixed_embeddings)) / len(fixed_embeddings)  # e.g. 15.6
threshold = HierarchicalTokenPooling.calibrate_distance_threshold(calibration_embeddings, target_tokens=target_tokens)
pooling = HierarchicalTokenPooling(distance_threshold=threshold)
```

This is how the calibrated row in the results table was measured, targeting 15.60 vectors overall. Those measurements use each dataset's complete corpus for calibration, while a representative sample can be used for your own index.
</details>
<br>

## Combining compression methods

Token pooling reduces the number of document vectors, while [Matryoshka training](../matryoshka/README.md) reduces their dimensionality and index quantization reduces the bits stored per coordinate. These choices address different parts of index storage and can be combined. Measure retrieval quality with the combination you intend to deploy, since each compression step can discard information needed to distinguish relevant documents.

## References

- [Learn to Pool](https://stefan-jo.github.io/learn-to-pool)
- [LightOn's pooling regularization study](https://huggingface.co/blog/lightonai/lateon-hpool-regularization)
- [MultiVectorTokenPoolingLoss API](../../../../docs/package_reference/multi_vector_encoder/losses.html#multivectortokenpoolingloss)

## See also

- [Matryoshka training](../matryoshka/README.md)
- [Fast-PLAID (index quantization)](https://github.com/lightonai/fast-plaid)
