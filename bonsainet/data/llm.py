
from typing import Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

import numpy as np
import random


class RandomTokens(Dataset):
    def __init__(self, tokenizer, seq_len, size=1_000, seed=None, **kwargs):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.size = size

        special_ids = set(
            [
                getattr(tokenizer, a)
                for a in [
                    "eos_token_id",
                    "pad_token_id",
                    "unk_token_id",
                ]
                if getattr(tokenizer, a, None) is not None
            ]
        )

        self.allowed_ids = torch.tensor(
            [i for i in range(tokenizer.vocab_size) if i not in special_ids]
        ).long()

        self._rand = torch.Generator()
        if seed is not None:
            self._rand.manual_seed(int(seed))

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        idxs = torch.randint(
            low=0,
            high=len(self.allowed_ids),
            size=(self.seq_len,),
            generator=self._rand,
            dtype=torch.long,
        )
        input_ids = self.allowed_ids[idxs]
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return input_ids, attention_mask


def get_llm_dataset(hub_path, split="train", **kwargs):
    path, name = hub_path.split("/")
    return load_dataset(path, name, split=split, **kwargs)


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids



# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(num_samples, seed, seq_len, tokenizer):
    """Prepare C4 language modeling data for training and validation.

    This function:
    - loads a small shard of the C4 dataset (train and validation);
    - draws `num_samples` random training sequences of length `seq_len`;
    - builds a training "loader" as a list of (input_ids, target_ids) pairs,
      where target tokens are masked with -100 except for the last position;
    - builds a long validation sequence wrapped in `TokenizerWrapper`.

    Returns a tuple `(train_samples, validation_wrapper)`.
    """

    # Load a shard of the C4 training split
    train_dataset = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    # Load a shard of the C4 validation split
    validation_dataset = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
        split="validation",
    )

    # Generate `num_samples` random training examples
    # Each example is a contiguous subsequence of length `seq_len`
    random.seed(seed)
    train_samples = []
    for _ in range(num_samples):
        # Keep sampling random documents until we get one long enough
        while True:
            random_doc_index = random.randint(0, len(train_dataset) - 1)
            tokenized_document = tokenizer(
                train_dataset[random_doc_index]["text"],
                return_tensors="pt",
            )
            # Only accept documents that are longer than `seq_len`
            if tokenized_document.input_ids.shape[1] > seq_len:
                break

        # Pick a random starting position within the document
        start_index = random.randint(
            0, tokenized_document.input_ids.shape[1] - seq_len - 1
        )
        end_index = start_index + seq_len

        # Slice out a window of `seq_len` tokens
        input_ids = tokenized_document.input_ids[:, start_index:end_index]

        # Targets are a shifted copy where all but the last token are ignored
        # (PyTorch cross-entropy uses -100 to mark tokens to be ignored.)
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        train_samples.append((input_ids, target_ids))

    # Build a long validation sequence by concatenating many validation texts
    concatenated_validation_text = " ".join(
        validation_dataset[:1100]["text"]
    )
    validation_tokenized = tokenizer(
        concatenated_validation_text,
        return_tensors="pt",
    )

    # Keep only the first 256 * seq_len tokens for validation
    validation_input_ids = validation_tokenized.input_ids[:, : (256 * seq_len)]

    # Wrap in a simple container that exposes `.input_ids`
    validation_wrapper = TokenizerWrapper(validation_input_ids)

    return train_samples, validation_wrapper


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
