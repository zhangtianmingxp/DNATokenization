"""Minimal local Caduceus tokenizer used by the original training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CaduceusTokenizer(PreTrainedTokenizer):
    """Character DNA tokenizer with the id layout expected by this repo."""

    def __init__(
        self,
        model_max_length: int,
        characters: Sequence[str] | None = None,
        padding_side: str = "left",
        **kwargs,
    ):
        self.characters = list(characters or ["A", "C", "G", "T", "N"])
        self.model_max_length = model_max_length

        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
        self.complement_map = {}
        for token, token_id in self._vocab_str_to_int.items():
            complement = complement_map[token] if token in complement_map else token
            self.complement_map[token_id] = self._vocab_str_to_int[complement]

        super().__init__(
            bos_token=bos_token,
            eos_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        result.setdefault("boundaries", [0] * len(result["input_ids"]))
        return result

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        result = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            result += token_ids_1 + [self.sep_token_id]
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        result = [0] * len([self.cls_token_id] + token_ids_0 + [self.sep_token_id])
        if token_ids_1 is not None:
            result += [1] * len(token_ids_1 + [self.sep_token_id])
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def get_config(self) -> Dict[str, object]:
        return {
            "characters": self.characters,
            "model_max_length": self.model_max_length,
        }

    def save_pretrained(self, save_directory, **kwargs):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "tokenizer_config.json").write_text(
            json.dumps(self.get_config(), indent=4),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        cfg = json.loads((Path(save_directory) / "tokenizer_config.json").read_text(encoding="utf-8"))
        return cls(**cfg)
