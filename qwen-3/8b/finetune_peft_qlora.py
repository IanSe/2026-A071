#!/usr/bin/env python3
"""
QLoRA: NF4 4-bit base + LoRA + optional 8-bit Adam (bitsandbytes).

Optimizer: set `TRAIN_OPTIM` (`adamw_torch` default; `paged_adamw_8bit` may fail on ROCm).
Configure via environment variables (see `docker-compose.yml`).
"""

from __future__ import annotations

import logging

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer

from power_telemetry import PowerTelemetry
from train_common import (
    TrainConfig,
    build_lora_config,
    build_sft_config,
    configure_distributed,
    init_seed,
    load_bioinstruct_datasets,
    load_train_config,
    load_tokenizer,
    local_rank,
    log_cuda_sanity,
    log_train_config,
    optional_hf_login,
    print_trainable_parameters,
    setup_logging,
)

_ALLOWED_OPTIMS = frozenset({"adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit"})


def _resolve_qlora_optim(cfg: TrainConfig) -> str:
    name = cfg.train_optim.strip()
    if name not in _ALLOWED_OPTIMS:
        raise ValueError(
            f"TRAIN_OPTIM must be one of {sorted(_ALLOWED_OPTIMS)}, got {cfg.train_optim!r}"
        )
    return name


def main() -> None:
    setup_logging()
    cfg = load_train_config()
    log_train_config(cfg)
    optim = _resolve_qlora_optim(cfg)
    if optim != "adamw_torch":
        logging.warning(
            "Using TRAIN_OPTIM=%s — if training fails on ROCm, set TRAIN_OPTIM=adamw_torch",
            optim,
        )

    optional_hf_login()
    init_seed(cfg.seed)
    configure_distributed()
    log_cuda_sanity()

    telemetry = PowerTelemetry(
        project_name='qwen3-8b-qlora',
        output_dir='./code_carbon_qwen3_8b',
    )
    telemetry.start()

    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)

    telemetry.begin_phase('dataset')
    train_ds, eval_ds = load_bioinstruct_datasets(
        tokenizer,
        cfg.dataset_id,
        cfg.dataset_split,
        cfg.eval_ratio,
        cfg.seed,
    )
    ds_energy = telemetry.end_phase('dataset')

    telemetry.begin_phase('load_model')
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    lr = local_rank()
    device_map = {"": lr} if lr >= 0 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = build_lora_config(cfg)
    training_args = build_sft_config(
        cfg,
        eval_ds,
        bf16=False,
        fp16=False,
        optim=optim,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    lmodel_energy = telemetry.end_phase('load_model')

    telemetry.begin_phase('fine_tuning')
    trainer.train()
    print_trainable_parameters(trainer.model)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    ft_energy = telemetry.end_phase('fine_tuning')

    telemetry.stop()
    print(f'Energy loading dataset: {ds_energy}')
    print(f'Energy loading model: {lmodel_energy}')
    print(f'Energy in fine-tuning model: {ft_energy}')


if __name__ == "__main__":
    main()
