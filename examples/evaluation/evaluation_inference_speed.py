"""
This examples measures the inference speed of a certain model

Usage:
python evaluation_inference_speed.py
OR
python evaluation_inference_speed.py model_name
"""

from sentence_transformers import SentenceTransformer
import sys
import time
import torch
from datasets import load_dataset

# Limit torch to 4 threads
torch.set_num_threads(4)


model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-nli-mean-tokens"

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)

max_sentences = 100000
all_nli_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
sentences = list(set(all_nli_dataset["anchor"]))[:max_sentences]

sentences = list(sentences)
print("Model Name:", model_name)
print("Number of sentences:", len(sentences))

for i in range(3):
    print("Run", i)
    start_time = time.time()
    emb = model.encode(sentences, batch_size=32)
    end_time = time.time()
    diff_time = end_time - start_time
    print("Done after {:.2f} seconds".format(diff_time))
    print("Speed: {:.2f} sentences / second".format(len(sentences) / diff_time))
    print("=====")
