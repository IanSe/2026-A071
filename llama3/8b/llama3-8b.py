import sys
import os
import subprocess

os.environ['CUDA_VISIBLE_DEVICES']="0,1"
import torch

pkgs = ['python-dotenv', 'trackio', 'transformers>=5.0', 'trl[peft]>=1.0.0', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'evaluate', 'bert_score']

def install_packages(packages):
    print("Resolving environment and installing packages via uv...")
    try:
        subprocess.check_call(['uv', 'pip', 'install'] + packages)
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

import json
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from accelerate import Accelerator
from trl import SFTTrainer, setup_chat_format
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge

gpus = get_gpus()
print(gpus)

accelerator = Accelerator()

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["instruction"] + " " + row["input"]},
               {"role": "assistant", "content": row["output"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


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

is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

if is_main:
    os.makedirs('code_carbon_bio-llama3-8b', exist_ok=True)

    log_name = "bio-llama3-8b-logs"
    _logger = logging.getLogger(log_name)
    _channel = logging.FileHandler(log_name + '.log')
    _logger.addHandler(_channel)
    _logger.setLevel(logging.INFO)

    start_http_server(8000)

    PHASE = Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase'])
    CPU_W = Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)')
    RAM_W = Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)')
    CPU_E = Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)')
    RAM_E = Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)')

    PHASES = ('dataset', 'load_model', 'fine_tuning')
    for _p in PHASES:
        PHASE.labels(phase=_p).set(0)

    POWER_CSV = './code_carbon_bio-llama3-8b/power_timeseries.csv'
    with open(POWER_CSV, 'w') as _f:
        _f.write('timestamp,cpu_w,ram_w,phase\n')

    _current_phase = {'name': 'idle'}


    class PromAndCsvLoggerOutput(LoggerOutput):
        """codecarbon LoggerOutput that ALSO updates Prometheus gauges and appends to power_timeseries.csv on every flush."""

        def _publish(self, total, delta):
            cpu_w = float(getattr(delta, 'cpu_power', 0.0) or 0.0)
            ram_w = float(getattr(delta, 'ram_power', 0.0) or 0.0)
            CPU_W.set(cpu_w)
            RAM_W.set(ram_w)
            CPU_E.set(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
            RAM_E.set(float(getattr(total, 'ram_energy', 0.0) or 0.0))
            try:
                with open(POWER_CSV, 'a') as fh:
                    fh.write(f"{datetime.utcnow().isoformat()},{cpu_w},{ram_w},{_current_phase['name']}\n")
            except Exception:
                pass

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
        project_name = 'bio-llama3-8b',
        output_dir="./code_carbon_bio-llama3-8b/",
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
    """Mark a fine-tuning phase active: set Prometheus gauge to 1 and start a Zeus window."""
    if is_main:
        _current_phase['name'] = name
    PHASE.labels(phase=name).set(1)
    monitor.begin_window(name)


def end_phase(name: str):
    """End a fine-tuning phase: close the Zeus window and clear the Prometheus gauge."""
    if is_main:
        energy = monitor.end_window(name)
    PHASE.labels(phase=name).set(0)
    _current_phase['name'] = 'idle'
    return energy

print('Mapping dataset')
begin_phase('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(format_chat_template, batched=True)
ds_energy = end_phase('dataset')

begin_phase('load_model')
model_name = "meta-llama/Llama-3.1-8B"
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(model_name)

compute_dtype = getattr(torch, "float16")


print('Downloading model')
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          dtype = torch.bfloat16,
          device_map={"": accelerator.local_process_index},
          attn_implementation="eager"
)

model, tokenizer = setup_chat_format(model, tokenizer)

dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

model.config.use_cache = False

training_arguments = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    eval_strategy="epoch",
    num_train_epochs=1,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    dataloader_num_workers = 4,
    log_level="info",
    save_steps=500,
    logging_steps=20,
    learning_rate=2e-5,
    warmup_steps=10,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=training_arguments,
        peft_config=peft_config,
)
lmodel_energy = end_phase('load_model')

begin_phase('fine_tuning')
trainer.train()

print_trainable_parameters(model)
ft_energy = end_phase('fine_tuning')

if is_main:
    new_model = 'Bio-llama3-8b'
    trainer.model.save_pretrained(new_model)
    model.push_to_hub("darmasrmz/bio-llama3-8b-lora", token = hf_token)
    tokenizer.push_to_hub("darmasrmz/bio-llama3-8b-lora", token = hf_token)


begin_phase('evaluation')
import evaluate
from tqdm import tqdm

bertscore = evaluate.load("bertscore")

dataset = load_dataset("bio-nlp-umass/bioinstruct", split="test")
dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

predictions = []
references = []

print("Starting generation...")
for row in tqdm(test_dataset):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    predictions.append(response)
    references.append(row['output'])

print("Calculating metrics...")
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
print(f"BERTScore F1 (Mean): {sum(bertscore_results['f1']) / len(bertscore_results['f1'])}")

evaluation_energy = end_phase('evaluation')

emissions = tracker.stop()

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

    try:
        power_df = pd.read_csv(POWER_CSV, parse_dates=['timestamp'])
        if not power_df.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(power_df['timestamp'], power_df['cpu_w'], label='CPU (W)')
            ax.plot(power_df['timestamp'], power_df['ram_w'], label='RAM (W)')

            phase_colors = {'dataset': 'tab:orange', 'load_model': 'tab:green', 'fine_tuning': 'tab:red'}
            for phase_name, color in phase_colors.items():
                mask = power_df['phase'] == phase_name
                if mask.any():
                    ax.axvspan(power_df.loc[mask, 'timestamp'].min(),
                            power_df.loc[mask, 'timestamp'].max(),
                            alpha=0.1, color=color, label=f'phase: {phase_name}')

            ax.set_title('Consumo de potencia del contenedor (CPU y RAM)')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Potencia (W)')
            ax.legend(loc='upper right')
            ax.grid(True)
            fig.tight_layout()
            fig.savefig('power_consumption.png', transparent=True)
            plt.close(fig)
    except FileNotFoundError:
        print(f'No power time series found at {POWER_CSV}')

    print(f'Energy loading dataset: {ds_energy}')
    print(f'Energy loading model: {lmodel_energy}')
    print(f'Energy in fine-tuning model: {ft_energy}')
    print(f'Energy in evaluation: {evaluation_energy}')