"""Synthetic DNA data and BERT-style nucleotide masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

DNA_PAD_ID = 0
DNA_ALPHABET: Dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
DNA_MASK_ID = 6
DNA_VOCAB_SIZE = 7


@dataclass(frozen=True)
class MaskingConfig:
    mask_probability: float = 0.15
    random_replace_probability: float = 0.10
    keep_original_probability: float = 0.10


class SyntheticDNADataset(Dataset[torch.Tensor]):
    """Tiny deterministic DNA-like dataset with motif-biased synthetic sequences."""

    def __init__(self, num_sequences: int = 256, seq_len: int = 128, seed: int = 0, n_probability: float = 0.02):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.n_probability = n_probability

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        bases = torch.randint(1, 5, (self.seq_len,), generator=generator)

        motif = torch.tensor([DNA_ALPHABET["A"], DNA_ALPHABET["C"], DNA_ALPHABET["G"], DNA_ALPHABET["T"]])
        stride = 17 + (index % 5)
        for start in range(index % stride, self.seq_len - len(motif) + 1, stride):
            bases[start : start + len(motif)] = motif

        n_mask = torch.rand(self.seq_len, generator=generator) < self.n_probability
        bases[n_mask] = DNA_ALPHABET["N"]
        return bases.long()


def mask_dna_batch(batch: List[torch.Tensor], config: MaskingConfig | None = None) -> Dict[str, torch.Tensor]:
    """Create masked inputs and MLM labels.

    Labels use ``-100`` for unmasked positions so ``torch.nn.CrossEntropyLoss`` ignores them.
    """

    if config is None:
        config = MaskingConfig()

    input_ids = torch.stack(batch, dim=0)
    labels = torch.full_like(input_ids, fill_value=-100)

    masked_positions = torch.rand(input_ids.shape, device=input_ids.device) < config.mask_probability
    labels[masked_positions] = input_ids[masked_positions]

    replacement_draw = torch.rand(input_ids.shape, device=input_ids.device)
    mask_token_positions = masked_positions & (
        replacement_draw >= config.random_replace_probability + config.keep_original_probability
    )
    random_positions = masked_positions & (replacement_draw < config.random_replace_probability)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_token_positions] = DNA_MASK_ID
    random_tokens = torch.randint(1, 6, input_ids.shape, device=input_ids.device)
    masked_input_ids[random_positions] = random_tokens[random_positions]

    return {"input_ids": masked_input_ids.long(), "labels": labels.long()}
