#!/usr/bin/env python3
"""
LoRA with an 8-bit bitsandbytes-quantized frozen base (not full QLoRA / NF4).

Requires a ROCm-compatible bitsandbytes build in the image.
Configure via environment variables (see `docker-compose.yml`).
"""

from __future__ import annotations

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer

from power_telemetry import PowerTelemetry
from train_common import (
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


def main() -> None:
    setup_logging()
    cfg = load_train_config()
    log_train_config(cfg)

    optional_hf_login()
    init_seed(cfg.seed)
    configure_distributed()
    log_cuda_sanity()

    telemetry = PowerTelemetry(
        project_name='qwen3-8b-int8',
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
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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
        optim="adamw_torch",
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
