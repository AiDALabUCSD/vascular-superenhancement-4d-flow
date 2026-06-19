"""Per-batch structured logging to attribute loss / grad-norm spikes to patients.

Training spikes (a batch with an unusually large ``loss_correction_mse`` or
``metric_grad_norm``) are usually driven by a few outlier samples. The ordinary
per-batch ``logger.info`` line records the losses but *not which patients were in
the batch*, so spikes can't be traced back to a culprit.

This callback writes one CSV row per training batch, tagged with the patient IDs
and timepoints in that batch, so spikes can be correlated with specific patients
post-hoc (e.g. ``sort -t, -k<grad_norm> batch_metrics/*.csv``).

DDP note: gradients are all-reduced before ``metric_grad_norm`` is measured, so
that value is *global* (identical across ranks). The per-component losses in
``outputs``, however, reflect each rank's *local* batch -- that's the signal that
attributes a spike to the patients that caused it. We therefore write one CSV
**per rank** and every rank logs (not just rank 0).

A live WARNING is also emitted when the watched metric exceeds a running
``mean + k * std``, so spikes surface in the train log as they happen (and the
row is flushed immediately, surviving a subsequent crash).
"""
from __future__ import annotations

import csv
import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .base_callback import Callback

logger = logging.getLogger(__name__)


def _as_float(v: Any) -> Optional[float]:
    try:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


class BatchMetricsCallback(Callback):
    """Log per-batch losses + patient IDs to a per-rank CSV for spike forensics."""

    def __init__(self, cfg: Any):
        bm = cfg.train.get("batch_metrics", None) or {}
        self.enabled: bool = bool(bm.get("enabled", True))
        # Metric watched for the live spike WARNING + the running statistics.
        self.spike_metric: str = str(bm.get("spike_metric", "loss_correction_mse"))
        self.spike_k: float = float(bm.get("spike_k", 5.0))
        self.spike_window: int = int(bm.get("spike_window", 100))
        self.spike_min_count: int = int(bm.get("spike_min_count", 20))

        self._fh = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None
        self._history: Deque[float] = deque(maxlen=self.spike_window)
        self._rank = 0

        if self.enabled:
            logger.info("BatchMetricsCallback initialized:")
            logger.info(f"  - spike_metric: {self.spike_metric} (k={self.spike_k}, window={self.spike_window})")

    def _ensure_writer(self, row: Dict[str, Any]) -> None:
        if self._writer is not None:
            return
        out_dir = Path.cwd() / "batch_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"train_batch_metrics_rank{self._rank}.csv"
        # Stable column order: metadata first, then the numeric metrics sorted.
        meta = ["epoch", "batch_idx", "global_step", "rank", "patient_ids",
                "time_indices", "vencs", "batch_size"]
        metric_keys = sorted(k for k in row if k not in meta)
        self._fieldnames = meta + metric_keys
        write_header = not path.exists()
        self._fh = path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
        logger.info(f"BatchMetricsCallback[rank {self._rank}] writing -> {path}")

    def on_train_batch_end(
        self, trainer: "BaseTrainer", batch: Any, batch_idx: int, outputs: Dict[str, Any]
    ) -> None:
        if not self.enabled:
            return
        self._rank = int(getattr(trainer, "rank", 0))

        pids = _as_list(batch.get("patient_id") if hasattr(batch, "get") else None)
        tidx = _as_list(batch.get("time_index") if hasattr(batch, "get") else None)
        vencs = _as_list(batch.get("venc") if hasattr(batch, "get") else None)

        row: Dict[str, Any] = {
            "epoch": int(getattr(trainer, "current_epoch", -1)),
            "batch_idx": batch_idx,
            "global_step": int(getattr(trainer, "global_step", -1)),
            "rank": self._rank,
            "patient_ids": ";".join(pids),
            "time_indices": ";".join(tidx),
            "vencs": ";".join(vencs),
            "batch_size": len(pids),
        }
        for key, value in outputs.items():
            if not (key.startswith("loss") or key.startswith("metric")):
                continue
            f = _as_float(value)
            if f is not None:
                row[key] = f

        self._ensure_writer(row)
        assert self._writer is not None and self._fh is not None
        self._writer.writerow(row)
        self._fh.flush()

        # --- Live spike detection on the watched metric ----------------------
        watched = row.get(self.spike_metric)
        if watched is None:
            return
        if len(self._history) >= self.spike_min_count:
            n = len(self._history)
            mean = sum(self._history) / n
            var = sum((x - mean) ** 2 for x in self._history) / n
            std = var ** 0.5
            if std > 0 and watched > mean + self.spike_k * std:
                logger.warning(
                    "BatchMetricsCallback[rank %d] SPIKE e%d b%d g%d: %s=%.4f "
                    "(running mean=%.4f std=%.4f, +%.1f sigma) patients=[%s] t=[%s]",
                    self._rank, row["epoch"], batch_idx, row["global_step"],
                    self.spike_metric, watched, mean, std,
                    (watched - mean) / std if std else float("nan"),
                    row["patient_ids"], row["time_indices"],
                )
        self._history.append(watched)

    def on_fit_end(self, trainer: "BaseTrainer") -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            self._writer = None
