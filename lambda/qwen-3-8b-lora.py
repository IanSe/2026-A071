import sys
import os
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

pkgs = ['python-dotenv', 'trackio', 'transformers', 'trl[peft]', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'pandas', 'matplotlib', 'prometheus-client', 'flash-attn']

def install_packages(packages):
    print("Resolving environment and installing packages via uv...")
    try:
        subprocess.check_call(['uv', 'pip', 'install'] + packages)
        subprocess.call(['uv', 'pip', 'uninstall', 'amdsmi'])
        print("All packages successfully installed and verified!")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")

install_packages(pkgs)

from dotenv import load_dotenv
from huggingface_hub import login, auth_list
import wandb

load_dotenv()
wandb_key = os.getenv('WANDB')
hf_token = os.getenv('HF_TOKEN')

wandb.login(key=wandb_key)
login(token=hf_token, add_to_git_credential=False)

auth_list()

import torch
import pynvml
from datetime import datetime, timezone
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge
from kernels import get_kernel

fa_module = get_kernel("kernels-community/flash-attn2", version=1)
flash_attn_func = fa_module.flash_attn_func

import transformers
import trl
print("Transformers: ", transformers.__version__)
print("TRL: ", trl.__version__)

is_main = True

gpus = get_gpus()
print(gpus)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


os.makedirs('code_carbon_qwen-3-8b-qlora', exist_ok=True)


if is_main:
    log_name = "bio-qwen-3-8b-qlora_logs"
    _logger = logging.getLogger(log_name)
    _channel = logging.FileHandler(log_name + '.log')
    _logger.addHandler(_channel)
    _logger.setLevel(logging.INFO)

    start_http_server(8000)

    PHASE = Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase'])
    CPU_W = Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)')
    RAM_W = Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)')
    GPU_W = Gauge('training_gpu_power_watts', 'GPU power estimate (watts)')
    CPU_E = Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)')
    RAM_E = Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)')
    GPU_E = Gauge('training_gpu_energy_kwh', 'Cumulative GPU energy from codecarbon (kWh)')

    PHASES = ('dataset', 'load_model', 'fine_tuning', 'evaluation')
    for _p in PHASES:
        PHASE.labels(phase=_p).set(0)

    POWER_CSV = './code_carbon_qwen-3-8b-qlora/power_timeseries.csv'
    with open(POWER_CSV, 'w') as _f:
        _f.write('timestamp,cpu_w,ram_w,cpu_e,ram_e,gpu_e,gpus,phase\n')
    PHASE_ENERGY_CSV = './code_carbon_qwen-3-8b-qlora/phase_energy.csv'
    with open(PHASE_ENERGY_CSV, 'w') as _f:
        _f.write('timestamp,phase,energy\n')

    _current_phase = {'name': 'idle'}
    class PromAndCsvLoggerOutput(LoggerOutput):
        """codecarbon LoggerOutput that ALSO updates Prometheus gauges and appends to power_timeseries.csv on every flush."""

        def _publish(self, total, delta):
            cpu_w = float(getattr(delta, 'cpu_power', 0.0) or 0.0)
            ram_w = float(getattr(delta, 'ram_power', 0.0) or 0.0)
            cpu_e = float(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
            ram_e = float(float(getattr(total, 'ram_energy', 0.0) or 0.0))
            CPU_W.set(cpu_w)
            RAM_W.set(ram_w)
            CPU_E.set(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
            RAM_E.set(float(getattr(total, 'ram_energy', 0.0) or 0.0))
            GPU_E.set(float(getattr(total, 'gpu_energy', 0.0) or 0.0))
            gpu_e = float(getattr(total, 'gpu_energy', 0.0) or 0.0)
            gpus_kw = 0.0
            GPU_W.set(gpus_kw)
            ts = datetime.now(timezone.utc).isoformat()
            with open(POWER_CSV, 'a') as fh:
                fh.write(f"{ts},{cpu_w},{ram_w},{cpu_e},{ram_e},{gpu_e},{gpus_kw},{_current_phase['name']}\n")

        def out(self, total, delta):
            super().out(total, delta)
            self._publish(total, delta)

        def live_out(self, total, delta):
            try:
                super().live_out(total, delta)
            except AttributeError:
                pass
            self._publish(total, delta)


    my_logger = PromAndCsvLoggerOutput(_logger, logging.INFO)

    tracker = EmissionsTracker(
        project_name = 'bio-qwen-3-8b-qlora',
        output_dir="./code_carbon_qwen-3-8b-qlora/",
        save_to_file=True,
        on_csv_write='append',
        output_file="emissions.csv",
        tracking_mode="process",
        measure_power_secs=1,
        save_to_logger=True,
        logging_logger=my_logger
    )
    tracker.start()

monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])


def begin_phase(name: str):
    """Mark a fine-tuning phase active: start a Zeus window on every rank, set Prometheus gauge on rank 0."""
    monitor.begin_window(name)
    if is_main:
        _current_phase['name'] = name
        PHASE.labels(phase=name).set(1)


def end_phase(name: str):
    """End a fine-tuning phase: close the Zeus window on every rank, clear Prometheus gauge on rank 0."""
    energy = monitor.end_window(name)
    if is_main:
        PHASE.labels(phase=name).set(0)
        _current_phase['name'] = 'idle'
        ts = datetime.now(timezone.utc).isoformat()
        _logger.info(f"phase={name} energy={energy}")
        with open(PHASE_ENERGY_CSV, 'a') as _f:
            _f.write(f"{ts},{name},{energy}\n")
    return energy


begin_phase('load_model')
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print('Downloading model')
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)

print('Mapping dataset')
begin_phase('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(formatting_prompts_func, batched=False)
ds_energy = end_phase('dataset')

dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.07,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)


training_arguments = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    eval_strategy="epoch",
    num_train_epochs=1,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    dataloader_num_workers = 4,
    bf16=True,
    log_level="info",
    save_steps=500,
    logging_steps=20,
    learning_rate=2e-5,
    warmup_steps=100,
    lr_scheduler_type="constant",
    push_to_hub=True,
    hub_token=hf_token,
    hub_model_id="darmasrmz/bio-qwen-3-8b-qlora",
    hub_strategy="end",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=training_arguments,
        peft_config=peft_config
)
lmodel_energy = end_phase('load_model')

begin_phase('fine_tuning')
trainer.train()

print_trainable_parameters(model)

ft_energy = end_phase('fine_tuning')

if is_main:
    tracker.stop()
    new_model = 'bio-qwen-3-8b-qlora'
    model.save_pretrained(new_model)
    model.push_to_hub("darmasrmz/bio-qwen-3-8b-qlora")
    tokenizer.save_pretrained(new_model)
    tokenizer.push_to_hub("darmasrmz/bio-qwen-3-8b-qlora")

if is_main:
    import pandas as pd
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history
    logs_df = pd.DataFrame(log_history)

    train_loss = logs_df.dropna(subset=['loss'])
    eval_loss = logs_df.dropna(subset=['eval_loss'])

    plt.figure(figsize=(15, 8))
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss')
    plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', marker='x')
    plt.title('Función de pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.tight_layout()
    plt.savefig('loss_function.png', transparent=True)

    print(f'Energy loading dataset: {ds_energy}')
    print(f'Energy loading model: {lmodel_energy}')
    print(f'Energy in fine-tuning model: {ft_energy}')

    energy_summary = pd.DataFrame([
        {'phase': 'dataset', 'energy': ds_energy},
        {'phase': 'load_model', 'energy': lmodel_energy},
        {'phase': 'fine_tuning', 'energy': ft_energy},
    ])
    energy_summary.to_csv('bio-qwen-3-8b-qlora-energy_summary.csv', index=False)