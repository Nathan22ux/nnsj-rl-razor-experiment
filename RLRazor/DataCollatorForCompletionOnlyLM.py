from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


@dataclass
class DataCollatorforCompletionOnlyLM:
    """
    Masks labels so loss is computed ONLY on completion tokens.

    Example:
      response_template = "Answer:"
      text = "Question: ... Answer: 42"
      -> labels for prompt tokens are -100, labels for completion are token ids
    """
    tokenizer: PreTrainedTokenizerBase
    response_template: str
    max_length: Optional[int] = None
    padding: str = "longest"
    truncation: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]

        # Encode full sequences
        enc = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length or self.tokenizer.model_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Build labels initialized to IGNORE
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)

        for i, text in enumerate(texts):
            if self.response_template not in text:
                # No template found -> skip loss entirely
                continue

            # Build prompt prefix up to response marker
            prompt = text.split(self.response_template)[0] + self.response_template

            # Tokenize prompt with same settings
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False
            ).input_ids

            # Labels only for completion tokens
            start = len(prompt_ids)
            labels[i, start:] = input_ids[i, start:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }