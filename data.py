from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


CHAT_DATASET_NAME = "OpenAssistant/oasst1"
MATH_DATASET_NAME = "gsm8k"
CODE_DATASET_NAME = "fka/CodeAlpaca-20k"


@dataclass
class DataConfig:
    max_length: int = 512
    shuffle_seed: int = 42
    chat_split: str = "train"
    math_split: str = "train"
    code_split: str = "train"
    validation_ratio: float = 0.02
    sample_size: Optional[int] = None


def _format_dialogue(user: str, assistant: str) -> str:
    return f"User: {user.strip()}\nAssistant: {assistant.strip()}"


def _load_chat_dataset(split: str, sample_size: Optional[int] = None) -> Dataset:
    raw = load_dataset(CHAT_DATASET_NAME, split=split, trust_remote_code=True)

    id_key = "id" if "id" in raw.column_names else "message_id"
    parent_key = "parent_id" if "parent_id" in raw.column_names else "parent_message_id"

    parents = {row[id_key]: row for row in raw if row.get(id_key) is not None}

    records: List[str] = []
    for row in raw:
        if row.get("role") != "assistant":
            continue
        parent_id = row.get(parent_key)
        if parent_id is None:
            continue
        parent = parents.get(parent_id)
        if not parent or parent.get("role") != "prompter":
            continue

        user_text = parent.get("text", "").strip()
        assistant_text = row.get("text", "").strip()
        if not user_text or not assistant_text:
            continue
        records.append(_format_dialogue(user_text, assistant_text))

    if sample_size:
        records = records[:sample_size]

    return Dataset.from_dict({"text": records})


def _load_math_dataset(split: str, sample_size: Optional[int] = None) -> Dataset:
    raw = load_dataset(MATH_DATASET_NAME, split=split)

    def generator() -> Iterable[str]:
        for row in raw:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            if not question or not answer:
                continue
            yield _format_dialogue(
                question,
                "Let's reason step by step.\n" + answer,
            )

    texts = list(generator())
    if sample_size:
        texts = texts[:sample_size]

    return Dataset.from_dict({"text": texts})


def _load_code_dataset(split: str, sample_size: Optional[int] = None) -> Dataset:
    raw = load_dataset(CODE_DATASET_NAME, split=split)

    def generator() -> Iterable[str]:
        for row in raw:
            instruction = row.get("instruction", "").strip()
            input_text = row.get("input", "").strip()
            output = row.get("output", "").strip()
            if not instruction and not input_text:
                continue
            user_parts = [instruction]
            if input_text:
                user_parts.append(input_text)
            user = "\n".join(user_parts)
            if not output:
                continue
            assistant = output
            yield _format_dialogue(user, assistant)

    texts = list(generator())
    if sample_size:
        texts = texts[:sample_size]

    return Dataset.from_dict({"text": texts})


def load_mixed_dataset(tokenizer: PreTrainedTokenizerBase, config: DataConfig) -> DatasetDict:
    chat_dataset = _load_chat_dataset(config.chat_split, config.sample_size)
    math_dataset = _load_math_dataset(config.math_split, config.sample_size)
    code_dataset = _load_code_dataset(config.code_split, config.sample_size)

    merged_train = concatenate_datasets([chat_dataset, math_dataset, code_dataset])

    validation_size = int(len(merged_train) * config.validation_ratio)
    if validation_size == 0:
        validation_size = min(1024, len(merged_train) // 100 + 1)

    shuffled = merged_train.shuffle(seed=config.shuffle_seed)
    validation_dataset = shuffled.select(range(validation_size))
    train_dataset = shuffled.select(range(validation_size, len(shuffled)))

    def tokenize(batch: dict[str, List[str]]) -> dict[str, List[List[int]]]:
        outputs = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_val = validation_dataset.map(tokenize, batched=True, remove_columns=["text"])

    return DatasetDict({
        "train": tokenized_train.shuffle(seed=config.shuffle_seed),
        "validation": tokenized_val.shuffle(seed=config.shuffle_seed),
    })
