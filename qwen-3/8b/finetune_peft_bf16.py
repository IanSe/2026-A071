#!/usr/bin/env python3
"""
PEFT (LoRA) on a BF16 Qwen3-8B base — no bitsandbytes weight quantization.

Configure via environment variables (see `train_common` and `docker-compose.yml`).
Multi-GPU (2x MI210): e.g. `accelerate launch --num_processes 2 finetune_peft_bf16.py`
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
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
        project_name='qwen3-8b-bf16',
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
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=cfg.trust_remote_code,
    )
    lr = local_rank()
    if torch.cuda.is_available():
        if lr >= 0:
            model = model.to(torch.device("cuda", lr))
        else:
            model = model.to("cuda")
    model.config.use_cache = False

    peft_config = build_lora_config(cfg)
    training_args = build_sft_config(
        cfg,
        eval_ds,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
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
