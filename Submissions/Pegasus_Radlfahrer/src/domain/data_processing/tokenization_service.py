from dataclasses import dataclass

from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    tokenized_dataset: Dataset
    vocab_size: int
    total_tokens: int


class DatasetTokenizationService:
    """Tokenizes a text dataset into token-id sequences."""

    def tokenize(
        self,
        dataset: Dataset,
        text_column: str,
        min_sequence_length: int,
    ) -> TokenizationResult:
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Text column '{text_column}' not found in dataset. Available columns: {dataset.column_names}"
            )

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = int(1e30) # As we are just tokenizing and not using GPT2 model we do not need to worry about context length at this stage.
        vocab_size = int(tokenizer.vocab_size)

        tokenized_rows: list[list[int]] = []
        total_tokens = 0
        for text in tqdm(dataset[text_column], total=dataset.num_rows, desc="tokenizing", unit="sample"):
            if not isinstance(text, str):
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < min_sequence_length:
                continue
            tokenized_rows.append(token_ids)
            total_tokens += len(token_ids)

        tokenized_dataset = Dataset.from_dict({"input_ids": tokenized_rows})
        return TokenizationResult(
            tokenized_dataset=tokenized_dataset,
            vocab_size=vocab_size,
            total_tokens=total_tokens,
        )
