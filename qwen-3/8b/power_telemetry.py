"""
In-process power telemetry for Qwen3-8B fine-tuning runs.

Publishes CPU + RAM power from codecarbon to Prometheus (HTTP :8000) and to a
per-second CSV, and tags every sample with the active fine-tuning phase.

GPU watts are NOT republished here: the compose stack already scrapes `amd_dme`
directly via ROCm SMI, which is authoritative. GPU energy per phase is still
captured via `ZeusMonitor` end-of-window summaries (logged, not a live gauge).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional


class PowerTelemetry:
    """Wires codecarbon + zeus + a prometheus_client HTTP server together.

    Usage:
        tele = PowerTelemetry(project_name='qwen3-8b', output_dir='./code_carbon', phases=('dataset', 'load_model', 'fine_tuning'))
        tele.start()
        tele.begin_phase('fine_tuning')
        ...
        energy = tele.end_phase('fine_tuning')
        tele.stop()
    """

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        phases: tuple[str, ...] = ('dataset', 'load_model', 'fine_tuning'),
        http_port: int = 8000,
        measure_power_secs: int = 1,
        log_name: Optional[str] = None,
    ) -> None:
        self.project_name = project_name
        self.output_dir = output_dir
        self.phases = phases
        self.http_port = http_port
        self.measure_power_secs = measure_power_secs
        self.log_name = log_name or f"{project_name}_power"
        self.power_csv = os.path.join(output_dir, 'power_timeseries.csv')
        self._current_phase = {'name': 'idle'}
        self._tracker = None
        self._monitor = None
        self._gauges = {}

    def start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        import torch
        from codecarbon import EmissionsTracker
        from codecarbon.output import LoggerOutput
        from prometheus_client import start_http_server, Gauge
        from zeus.monitor import ZeusMonitor

        start_http_server(self.http_port)

        self._gauges = {
            'phase': Gauge('training_phase', 'Current fine-tuning phase (1=active, 0=inactive)', ['phase']),
            'cpu_w': Gauge('training_cpu_power_watts', 'CPU power from codecarbon (watts)'),
            'ram_w': Gauge('training_ram_power_watts', 'RAM power from codecarbon, estimate (watts)'),
            'cpu_e': Gauge('training_cpu_energy_kwh', 'Cumulative CPU energy (kWh)'),
            'ram_e': Gauge('training_ram_energy_kwh', 'Cumulative RAM energy (kWh)'),
        }
        for p in self.phases:
            self._gauges['phase'].labels(phase=p).set(0)

        with open(self.power_csv, 'w') as f:
            f.write('timestamp,cpu_w,ram_w,phase\n')

        cc_logger = logging.getLogger(self.log_name)
        cc_logger.addHandler(logging.FileHandler(self.log_name + '.log'))
        cc_logger.setLevel(logging.INFO)

        telemetry = self

        class PromAndCsvLoggerOutput(LoggerOutput):
            """codecarbon LoggerOutput that updates Prometheus gauges and appends to the power CSV."""

            def _publish(self, total, delta):
                cpu_w = float(getattr(delta, 'cpu_power', 0.0) or 0.0)
                ram_w = float(getattr(delta, 'ram_power', 0.0) or 0.0)
                telemetry._gauges['cpu_w'].set(cpu_w)
                telemetry._gauges['ram_w'].set(ram_w)
                telemetry._gauges['cpu_e'].set(float(getattr(total, 'cpu_energy', 0.0) or 0.0))
                telemetry._gauges['ram_e'].set(float(getattr(total, 'ram_energy', 0.0) or 0.0))
                try:
                    with open(telemetry.power_csv, 'a') as fh:
                        fh.write(
                            f"{datetime.utcnow().isoformat()},{cpu_w},{ram_w},{telemetry._current_phase['name']}\n"
                        )
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

        my_logger = PromAndCsvLoggerOutput(cc_logger, logging.INFO)

        self._tracker = EmissionsTracker(
            project_name=self.project_name,
            output_dir=self.output_dir,
            save_to_file=True,
            on_csv_write='append',
            output_file='emissions.csv',
            tracking_mode='process',
            measure_power_secs=self.measure_power_secs,
            save_to_logger=True,
            logging_logger=my_logger,
        )
        self._tracker.start()

        try:
            gpu_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self._monitor = ZeusMonitor(gpu_indices=[gpu_index])
        except Exception as exc:
            logging.warning('ZeusMonitor init failed (%s); per-phase GPU energy disabled', exc)
            self._monitor = None

    def begin_phase(self, name: str) -> None:
        """Mark a fine-tuning phase active: set Prometheus gauge to 1 and start a Zeus window."""
        self._current_phase['name'] = name
        if self._gauges:
            self._gauges['phase'].labels(phase=name).set(1)
        if self._monitor is not None:
            try:
                self._monitor.begin_window(name)
            except Exception as exc:
                logging.warning('ZeusMonitor begin_window(%s) failed: %s', name, exc)

    def end_phase(self, name: str):
        """End a fine-tuning phase: close the Zeus window and clear the Prometheus gauge."""
        energy = None
        if self._monitor is not None:
            try:
                energy = self._monitor.end_window(name)
            except Exception as exc:
                logging.warning('ZeusMonitor end_window(%s) failed: %s', name, exc)
        if self._gauges:
            self._gauges['phase'].labels(phase=name).set(0)
        self._current_phase['name'] = 'idle'
        return energy

    def stop(self):
        """Stop the codecarbon tracker and return aggregate emissions (or None)."""
        if self._tracker is None:
            return None
        try:
            return self._tracker.stop()
        except Exception as exc:
            logging.warning('codecarbon tracker.stop() failed: %s', exc)
            return None
