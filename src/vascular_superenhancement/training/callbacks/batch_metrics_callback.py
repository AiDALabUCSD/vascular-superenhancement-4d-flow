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
import math
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
        # Metrics watched for the live spike WARNING (each gets its own running
        # window). ``spike_metrics`` (list) is preferred; ``spike_metric`` (str)
        # is still honoured for backward compatibility.
        metrics_cfg = bm.get("spike_metrics", None)
        if metrics_cfg:
            self.spike_metrics: List[str] = [str(m) for m in metrics_cfg]
        else:
            single = bm.get("spike_metric", None)
            self.spike_metrics = [str(single)] if single else [
                "loss_correction_mse",   # weighted correction term
                "loss_generator",        # total (catches cine-side spikes)
                "metric_grad_norm",      # the actual update magnitude
                "loss_correction_mse_vz",  # through-plane: most flow-relevant
            ]
        self.spike_k: float = float(bm.get("spike_k", 5.0))
        self.spike_window: int = int(bm.get("spike_window", 100))
        self.spike_min_count: int = int(bm.get("spike_min_count", 20))
        # Suppress the NaN/Inf alarm during early AMP GradScaler warmup, where a
        # few inf grad-norms are expected and benign.
        self.nan_alarm_after_step: int = int(bm.get("nan_alarm_after_step", 50))
        # Used to flag when gradient clipping actually engaged.
        self.grad_clip_max_norm: float = float(cfg.train.get("grad_clip_max_norm", 0.0) or 0.0)

        self._fh = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None
        self._history: Dict[str, Deque[float]] = {
            m: deque(maxlen=self.spike_window) for m in self.spike_metrics
        }
        self._rank = 0

        if self.enabled:
            logger.info("BatchMetricsCallback initialized:")
            logger.info(f"  - spike_metrics: {self.spike_metrics} (k={self.spike_k}, window={self.spike_window})")
            logger.info(f"  - nan_alarm_after_step: {self.nan_alarm_after_step}, grad_clip_max_norm: {self.grad_clip_max_norm}")

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

        gstep = row["global_step"]

        # --- NaN/Inf hard alarm (independent of the rolling statistics) -------
        # Fires immediately on any non-finite loss/grad, since a NaN/Inf mid-run
        # means divergence and must surface at once. Gated past AMP warmup.
        if gstep >= self.nan_alarm_after_step:
            bad = [k for k, v in row.items()
                   if (k.startswith("loss") or k.startswith("metric"))
                   and isinstance(v, float) and not math.isfinite(v)]
            if bad:
                logger.warning(
                    "BatchMetricsCallback[rank %d] NON-FINITE e%d b%d g%d: %s patients=[%s] t=[%s]",
                    self._rank, row["epoch"], batch_idx, gstep,
                    ",".join(bad), row["patient_ids"], row["time_indices"],
                )

        # --- Live spike detection on each watched metric ---------------------
        for metric in self.spike_metrics:
            watched = row.get(metric)
            if watched is None or not (isinstance(watched, float) and math.isfinite(watched)):
                continue
            hist = self._history[metric]
            if len(hist) >= self.spike_min_count:
                n = len(hist)
                mean = sum(hist) / n
                std = (sum((x - mean) ** 2 for x in hist) / n) ** 0.5
                if std > 0 and watched > mean + self.spike_k * std:
                    clipped = (
                        " CLIPPED" if metric == "metric_grad_norm"
                        and self.grad_clip_max_norm > 0
                        and watched > self.grad_clip_max_norm else ""
                    )
                    logger.warning(
                        "BatchMetricsCallback[rank %d] SPIKE[%s%s] e%d b%d g%d: %.4f "
                        "(running mean=%.4f std=%.4f, +%.1f sigma) patients=[%s] t=[%s]",
                        self._rank, metric, clipped, row["epoch"], batch_idx, gstep,
                        watched, mean, std, (watched - mean) / std,
                        row["patient_ids"], row["time_indices"],
                    )
            hist.append(watched)

    def on_fit_end(self, trainer: "BaseTrainer") -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            self._writer = None
