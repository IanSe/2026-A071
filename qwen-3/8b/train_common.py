"""
Shared helpers for Qwen3-8B SFT (bioinstruct formatting, LoRA, env-based config).

Configuration is read from environment variables so docker-compose / `.env` can drive
training without CLI flags. Set variables on the `qwen-3-fine-tuning` service or in a
`.env` file next to `docker-compose.yml`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_DATASET_ID = "bio-nlp-umass/bioinstruct"

# Env keys (single source of truth for docs / compose)
ENV_MODEL_ID = "MODEL_ID"
ENV_DATASET_ID = "DATASET_ID"
ENV_DATASET_SPLIT = "DATASET_SPLIT"
ENV_OUTPUT_DIR = "OUTPUT_DIR"
ENV_MAX_SEQ_LENGTH = "MAX_SEQ_LENGTH"
ENV_NUM_TRAIN_EPOCHS = "NUM_TRAIN_EPOCHS"
ENV_LEARNING_RATE = "LEARNING_RATE"
ENV_LORA_R = "LORA_R"
ENV_LORA_ALPHA = "LORA_ALPHA"
ENV_LORA_DROPOUT = "LORA_DROPOUT"
ENV_PER_DEVICE_TRAIN_BATCH_SIZE = "PER_DEVICE_TRAIN_BATCH_SIZE"
ENV_PER_DEVICE_EVAL_BATCH_SIZE = "PER_DEVICE_EVAL_BATCH_SIZE"
ENV_GRADIENT_ACCUMULATION_STEPS = "GRADIENT_ACCUMULATION_STEPS"
ENV_WARMUP_STEPS = "WARMUP_STEPS"
ENV_LOGGING_STEPS = "LOGGING_STEPS"
ENV_SAVE_STEPS = "SAVE_STEPS"
ENV_SEED = "SEED"
ENV_EVAL_RATIO = "EVAL_RATIO"
ENV_REPORT_TO = "REPORT_TO"
ENV_TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"
ENV_TRAIN_OPTIM = "TRAIN_OPTIM"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

VRAM_GUIDANCE = (
    "VRAM (2x AMD MI210, ~64 GB HBM2 each): BF16 LoRA is usually safe with "
    "per_device_train_batch_size=1, gradient_checkpointing, and max_seq_length "
    "2048–4096; int8 and QLoRA reduce weight memory. Increase sequence length or "
    "batch only after monitoring rocm-smi / torch memory."
)


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class TrainConfig:
    model_id: str
    dataset_id: str
    dataset_split: str
    output_dir: str
    max_seq_length: int
    num_train_epochs: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    logging_steps: int
    save_steps: int
    seed: int
    eval_ratio: float
    report_to: str
    trust_remote_code: bool
    train_optim: str


def load_train_config() -> TrainConfig:
    """Load training options from environment (defaults suit MI210 + docker-compose)."""
    return TrainConfig(
        model_id=_env_str(ENV_MODEL_ID, DEFAULT_MODEL_ID),
        dataset_id=_env_str(ENV_DATASET_ID, DEFAULT_DATASET_ID),
        dataset_split=_env_str(ENV_DATASET_SPLIT, "train"),
        output_dir=_env_str(ENV_OUTPUT_DIR, "./outputs"),
        max_seq_length=_env_int(ENV_MAX_SEQ_LENGTH, 2048),
        num_train_epochs=_env_int(ENV_NUM_TRAIN_EPOCHS, 1),
        learning_rate=_env_float(ENV_LEARNING_RATE, 2e-5),
        lora_r=_env_int(ENV_LORA_R, 16),
        lora_alpha=_env_int(ENV_LORA_ALPHA, 32),
        lora_dropout=_env_float(ENV_LORA_DROPOUT, 0.05),
        per_device_train_batch_size=_env_int(ENV_PER_DEVICE_TRAIN_BATCH_SIZE, 1),
        per_device_eval_batch_size=_env_int(ENV_PER_DEVICE_EVAL_BATCH_SIZE, 1),
        gradient_accumulation_steps=_env_int(ENV_GRADIENT_ACCUMULATION_STEPS, 8),
        warmup_steps=_env_int(ENV_WARMUP_STEPS, 100),
        logging_steps=_env_int(ENV_LOGGING_STEPS, 20),
        save_steps=_env_int(ENV_SAVE_STEPS, 500),
        seed=_env_int(ENV_SEED, 42),
        eval_ratio=_env_float(ENV_EVAL_RATIO, 0.2),
        report_to=_env_str(ENV_REPORT_TO, "none"),
        trust_remote_code=_env_bool(ENV_TRUST_REMOTE_CODE, False),
        train_optim=_env_str(ENV_TRAIN_OPTIM, "adamw_torch"),
    )


def log_train_config(cfg: TrainConfig) -> None:
    logging.info("Train config: %s", VRAM_GUIDANCE)
    logging.info("Train config: %s", asdict(cfg))


def optional_hf_login() -> None:
    try:
        from huggingface_hub import login
    except ImportError:
        return
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def load_tokenizer(model_id: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _row_to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    instruction = example["instruction"]
    raw_inp = example.get("input")
    if isinstance(raw_inp, str) and raw_inp.strip():
        user_content = f"{instruction}\n{raw_inp.strip()}"
    else:
        user_content = instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def format_bioinstruct_batch(
    batch: dict[str, list[Any]],
    tokenizer: AutoTokenizer,
) -> dict[str, list[str]]:
    n = len(batch["instruction"])
    texts: list[str] = []
    for i in range(n):
        ex = {k: batch[k][i] for k in batch}
        messages = _row_to_messages(ex)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = (
                f"### User:\n{messages[0]['content']}\n\n"
                f"### Assistant:\n{messages[1]['content']}"
            )
        texts.append(text)
    return {"text": texts}


def load_bioinstruct_datasets(
    tokenizer: AutoTokenizer,
    dataset_id: str,
    split: str,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    raw = load_dataset(dataset_id, split=split)
    cols = raw.column_names
    mapped = raw.map(
        lambda b: format_bioinstruct_batch(b, tokenizer),
        batched=True,
        remove_columns=cols,
    )
    if eval_ratio and eval_ratio > 0:
        parts = mapped.train_test_split(test_size=eval_ratio, seed=seed)
        return parts["train"], parts["test"]
    return mapped, None


def build_lora_config(cfg: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )


def build_sft_config(
    cfg: TrainConfig,
    eval_ds: Dataset | None,
    *,
    bf16: bool | None = None,
    fp16: bool = False,
    optim: str = "adamw_torch",
):
    """Build TRL `SFTConfig` (uses `max_length`, not deprecated `max_seq_length`)."""
    from trl import SFTConfig

    if bf16 is None:
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    eval_strategy = "epoch" if eval_ds is not None else "no"
    return SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=cfg.report_to,
        seed=cfg.seed,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        optim=optim,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_fn = getattr(model, "num_parameters", None)
    if callable(num_fn):
        total = int(num_fn())
    else:
        total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    logging.info(
        "trainable params: %s || all params: %s || trainable%%: %.4f",
        trainable,
        total,
        pct,
    )


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )


def init_seed(seed: int) -> None:
    set_seed(seed)


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def configure_distributed() -> None:
    lr = local_rank()
    if lr >= 0:
        torch.cuda.set_device(lr)


def log_cuda_sanity() -> None:
    if not torch.cuda.is_available():
        logging.warning("CUDA/HIP not available; training will fail for GPU scripts.")
        return
    logging.info("Device: %s", torch.cuda.get_device_name(0))
