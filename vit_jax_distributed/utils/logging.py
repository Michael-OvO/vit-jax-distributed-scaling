"""Metrics logging utilities for distributed training experiments."""

import csv
import json
import sys
from datetime import datetime


class MetricsLogger:
    """Collects and exports training metrics across steps and epochs."""

    def __init__(self, config=None, num_devices=None, **metadata):
        self.step_logs = []
        self.epoch_logs = []
        self.metadata = {
            "config": config if isinstance(config, dict) else None,
            "num_devices": num_devices,
            "start_time": datetime.now().isoformat(),
            **metadata,
        }

    def log_step(self, step, metrics_dict):
        """Record metrics for a single training step."""
        entry = {"step": step, **metrics_dict}
        self.step_logs.append(entry)

    def log_epoch(self, epoch, metrics_dict):
        """Record metrics for a completed epoch."""
        entry = {"epoch": epoch, **metrics_dict}
        self.epoch_logs.append(entry)

    def save_csv(self, path, which="step"):
        """Save step or epoch logs to a CSV file.

        Args:
            path: File path for the CSV output.
            which: ``"step"`` or ``"epoch"`` to select the log source.
        """
        logs = self.step_logs if which == "step" else self.epoch_logs
        if not logs:
            return
        fieldnames = list(logs[0].keys())
        # Collect any extra keys that appear in later entries.
        for entry in logs:
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    def save_json(self, path):
        """Save all logs and metadata to a JSON file."""
        data = {
            "metadata": self.metadata,
            "step_logs": self.step_logs,
            "epoch_logs": self.epoch_logs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self, file=None):
        """Print a human-readable summary table to stdout (or *file*)."""
        out = file or sys.stdout
        out.write("\n=== Training Summary ===\n")

        if self.metadata.get("config"):
            out.write("\nConfig:\n")
            for k, v in self.metadata["config"].items():
                out.write(f"  {k}: {v}\n")
        if self.metadata.get("num_devices") is not None:
            out.write(f"\nDevices: {self.metadata['num_devices']}\n")

        if self.epoch_logs:
            out.write("\nEpoch metrics:\n")
            headers = list(self.epoch_logs[0].keys())
            col_widths = {h: max(len(h), 10) for h in headers}
            header_line = "  ".join(h.rjust(col_widths[h]) for h in headers)
            out.write(f"  {header_line}\n")
            out.write(f"  {'-' * len(header_line)}\n")
            for entry in self.epoch_logs:
                row = "  ".join(
                    _fmt(entry.get(h, "")).rjust(col_widths[h]) for h in headers
                )
                out.write(f"  {row}\n")

        if self.step_logs:
            out.write(f"\nTotal steps logged: {len(self.step_logs)}\n")

        out.write("========================\n\n")

    def summary_dict(self):
        """Return a summary dict describing the full run."""
        summary = {
            "metadata": self.metadata,
            "total_steps": len(self.step_logs),
            "total_epochs": len(self.epoch_logs),
        }
        if self.epoch_logs:
            last = self.epoch_logs[-1]
            summary["final_epoch_metrics"] = last
        if self.step_logs:
            summary["first_step"] = self.step_logs[0]
            summary["last_step"] = self.step_logs[-1]
        summary["end_time"] = datetime.now().isoformat()
        return summary


def _fmt(value):
    """Format a value for the summary table."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
