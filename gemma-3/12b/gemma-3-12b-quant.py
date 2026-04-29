import sys
import os
import subprocess

os.environ['CUDA_VISIBLE_DEVICES']="0,1"
import torch

pkgs = ['python-dotenv', 'trackio', 'transformers>=5.0', 'trl[peft]>=1.0.0', 'accelerate', 'bitsandbytes', 'datasets', 'ninja', 'codecarbon', 'packaging', 'zeus', 'wandb', 'huggingface-hub', 'tqdm', 'evaluate', 'nltk', 'pandas', 'matplotlib', 'prometheus-client']

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
from datetime import datetime, timezone
from datasets import load_dataset
import urllib.request
import threading
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from accelerate import Accelerator
from trl import SFTTrainer
from codecarbon import EmissionsTracker
from codecarbon.output import LoggerOutput
import logging
from zeus.device import get_gpus
from zeus.monitor import ZeusMonitor
from prometheus_client import start_http_server, Gauge

import transformers
import trl
import accelerate
print("Transformers: ", transformers.__version__)
print("TRL: ", trl.__version__)
print("Accelerate: ", accelerate.__version__)

is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

gpus = get_gpus()
print(gpus)

accelerator = Accelerator()

def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["instruction"] + " " + sample["input"]},
          {"role": "assistant", "content": sample["output"]}
      ]
  }


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


os.makedirs('code_carbon_gemma_3_12b_quant', exist_ok=True)

tracker = None

if is_main:
    log_name = "biogemma_3_12b_quant_logs"
    _logger = logging.getLogger(log_name)
    _channel = logging.FileHandler(log_name + '.log')
    _logger.addHandler(_channel)
    _logger.setLevel(logging.INFO)

    start_http_server(8000)

    PHASE = Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase'])
    CPU_W = Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)')
    RAM_W = Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)')
    GPU_W = Gauge('training_gpu_power_watts', 'GPU power total from amd_dme (watts)')
    CPU_E = Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)')
    RAM_E = Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)')
    GPU_E = Gauge('training_gpu_energy_kwh', 'Cumulative GPU energy from codecarbon (kWh)')

    PHASES = ('dataset', 'load_model', 'fine_tuning', 'evaluation')
    for _p in PHASES:
        PHASE.labels(phase=_p).set(0)

    POWER_CSV = './code_carbon_gemma_3_12b_quant/power_timeseries.csv'
    with open(POWER_CSV, 'w') as _f:
        _f.write('timestamp,cpu_w,ram_w,cpu_e,ram_e,gpu_e,gpus,phase\n')

    _current_phase = {'name': 'idle'}
    PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://prometheus:9090')
    AMD_DME_URL = os.environ.get('AMD_DME_URL', 'http://amd_dme:5000/metrics')
    _gpu_energy_re = re.compile(
        r'^gpu_energy_consumed\{[^}]*\}\s+([0-9.eE+-]+)', re.MULTILINE
    )

    def query_prom(promql: str, timeout: float = 2.0):
        """Run an instant PromQL query, return list of (labels_dict, value_float)."""
        url = f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(promql)}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                payload = json.loads(r.read().decode('utf-8'))
            if payload.get('status') != 'success':
                return []
            return [
                (item['metric'], float(item['value'][1]))
                for item in payload['data']['result']
            ]
        except Exception as e:
            print(f"[prom] query failed: {e}", flush=True)
            return []

    def _read_gpu_power_w():
        """Total GPU power across all hosts, watts."""
        rows = query_prom('sum by (hostname) (rate(gpu_energy_consumed[2m])) / 1e6')
        return sum(v for _, v in rows)

    def _read_amd_gpu_energy_kwh():
        try:
            with urllib.request.urlopen(AMD_DME_URL, timeout=2) as r:
                text = r.read().decode('utf-8', errors='ignore')
            uj = sum(float(m.group(1)) for m in _gpu_energy_re.finditer(text))
            return uj / 3.6e15
        except Exception:
            return 0.0

    def _gpu_power_loop(stop_evt):
        while not stop_evt.is_set():
            GPU_E.set(_read_amd_gpu_energy_kwh())
            stop_evt.wait(1.0)

    _gpu_stop = threading.Event()
    _gpu_thread = threading.Thread(target=_gpu_power_loop, args=(_gpu_stop,), daemon=True)
    _gpu_thread.start()


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
            gpu_e = _read_amd_gpu_energy_kwh()
            gpus_kw = _read_gpu_power_w()
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
        project_name = 'bio-gemma-3-12b-quant',
        output_dir="./code_carbon_gemma_3_12b_quant/",
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
    return energy

print('Mapping dataset')
begin_phase('dataset')
dataset = load_dataset("bio-nlp-umass/bioinstruct", split="train")
dataset = dataset.map(create_conversation, remove_columns=[], batched=False)
ds_energy = end_phase('dataset')

begin_phase('load_model')
model_name = "google/gemma-3-12b-it"
device = accelerator.device
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=compute_dtype,
)

print('Downloading model')
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=bnb_config,
          dtype=torch.bfloat16,
          device_map={"": accelerator.local_process_index},
          attn_implementation="eager"
)

dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset_splits['train']
test_dataset = dataset_splits['test']

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

training_arguments = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    eval_strategy="epoch",
    num_train_epochs=1,
    optim="adamw_torch_fused",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
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
    hub_model_id="darmasrmz/bio-gemma-3-12b-quant",
    hub_strategy="end",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=training_arguments,
)
lmodel_energy = end_phase('load_model')

begin_phase('fine_tuning')
trainer.train()

print_trainable_parameters(model)


ft_energy = end_phase('fine_tuning')

if is_main:
    new_model = 'Bio-gemma-3-12b-quant'
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

accelerator.wait_for_everyone()

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
    energy_summary.to_csv('bio-gemma-3-12b-quant-energy_summary.csv', index=False)
