"""In-loop aortic / pulmonary flow validation for the dual-task model.

At a configurable cadence this callback reconstructs, on the downsampled grid the
model actually operates on, the model-corrected velocity field for a handful of
validation patients (running the generator over every cardiac timepoint), then
measures aortic flow, pulmonary flow, and Qp:Qs using the precomputed auto-flow
geometry. It logs three variants per patient -- ``uncorrected``, ``gt`` (the
ground-truth phase-corrected field) and ``model`` -- so you can watch whether the
model's flow estimates move toward the GT-corrected reference as training
progresses.

Everything is gated behind ``cfg.train.flow_validation.enabled`` (default off) and
only runs on the main process; patients whose auto-flow geometry has not been
precomputed are skipped gracefully.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.amp.autocast_mode import autocast

from .base_callback import Callback
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.training.datasets import get_downsampled_mag_keys
from vascular_superenhancement.flow_eval.geometry_cache import PatientFlowGeometry
from vascular_superenhancement.flow_eval.validation import (
    build_downsampled_cache,
    load_downsampled_mag_frames,
    load_downsampled_velocity,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

_VARIANTS = ("uncorrected", "gt", "model")


def _key_to_offset(key: str) -> int:
    """Map a downsampled mag subject key to its temporal offset from the centre."""
    if key == "mag_center":
        return 0
    if key.startswith("mag_offset_n"):
        return -int(key[len("mag_offset_n"):])
    if key.startswith("mag_offset_p"):
        return int(key[len("mag_offset_p"):])
    raise ValueError(f"unexpected mag key: {key}")


def _rescale01(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max rescale to [0, 1] (matches tio.RescaleIntensity default)."""
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


class FlowValidationCallback(Callback):
    """Measure Ao / PA / Qp:Qs for validation patients during training."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        fv = cfg.train.get("flow_validation", None) or {}

        self.enabled = bool(fv.get("enabled", False))
        if cfg.train.get("trainer_type", "gan") != "dual_task":
            if self.enabled:
                logger.warning(
                    "flow_validation requested but trainer_type != dual_task; disabling."
                )
            self.enabled = False

        self.every_n_epochs = max(1, int(fv.get("every_n_epochs", 10)))
        self.num_patients = int(fv.get("num_patients", 8))  # -1 = all val patients
        self.explicit_ids: List[str] = list(fv.get("patient_ids", []) or [])
        # Patients with known-broken auto-flow geometry (e.g. flipped plane normals
        # -> negative flow) are still measured and written to the CSV, but excluded
        # from the *aggregate* metrics so they don't corrupt the cohort means/MAE.
        self.exclude_ids: List[str] = list(fv.get("exclude_patient_ids", []) or [])
        self.seg_threshold = float(fv.get("seg_threshold", 0.0))
        self.run_on_first_epoch = bool(fv.get("run_on_first_epoch", True))
        self.cache_uncorrected_in_ram = bool(fv.get("cache_uncorrected_in_ram", True))
        # Number of cardiac timepoints to run through the generator per forward
        # pass during measurement. Batching avoids the per-timepoint launch/sync
        # overhead that otherwise leaves the GPU idle. ~training batch size.
        self.tp_chunk = max(1, int(fv.get("timepoint_chunk", 10)))

        self.log_to_wandb = bool(fv.get("log_to_wandb", True)) and bool(cfg.wandb.enabled)
        self.write_csv = bool(fv.get("csv", True))

        self.downsampled_folder = cfg.data.downsampled_folder
        self.temporal_mag_offsets = list(cfg.train.temporal_mag_offsets)
        self.mag_keys = get_downsampled_mag_keys(self.temporal_mag_offsets)
        # Per-channel temporal offset for each magnitude input channel (centre = 0),
        # so we can assemble the model input by indexing preloaded frames instead of
        # rebuilding a TorchIO subject (and re-reading disk) per timepoint.
        self._mag_offsets = [_key_to_offset(k) for k in self.mag_keys]
        self.use_amp = bool(cfg.train.get("use_amp", False)) and torch.cuda.is_available()

        self.config_name = cfg.path_config.path_config_name
        self.output_dir = Path.cwd() / "flow_validation"

        # DDP: only rank 0 measures flow (the per-patient cost is ~2s when a rank
        # runs uncontended, but two ranks measuring at once collide on CPU/PCIe and
        # blow up ~10x). The other ranks block on a gloo (CPU) barrier so they
        # *sleep* instead of racing into the next training epoch -- otherwise their
        # training dataloaders/forward passes would starve rank 0's measurement.
        self.rank = 0
        self.world_size = 1
        self._gather_group = None

        # Resolved at on_fit_start: patient_id -> Patient, and cached geometry.
        self._patients: Dict[str, Patient] = {}
        self._geometry: Dict[str, Optional[PatientFlowGeometry]] = {}
        self._venc: Dict[str, float] = {}
        # Static (training-invariant) baseline flow numbers, computed once.
        self._baseline: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Optional in-RAM caches of the (static, training-invariant) model inputs,
        # gated by ``cache_uncorrected_in_ram``. Without these, every flow epoch
        # re-reads ~80 cold gzip NIfTI frames per patient (60 velocity + 20
        # magnitude); caching them makes every epoch after the first nearly pure
        # GPU (~2s/patient).
        self._uncorrected_cache: Dict[str, np.ndarray] = {}
        self._mag_cache: Dict[str, np.ndarray] = {}

        if self.enabled:
            logger.info("FlowValidationCallback initialized:")
            logger.info(f"  - every_n_epochs: {self.every_n_epochs}")
            logger.info(
                f"  - patients: {self.explicit_ids or f'first {self.num_patients} of val split'}"
            )
            logger.info(f"  - log_to_wandb: {self.log_to_wandb}, csv: {self.write_csv}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: "BaseTrainer") -> None:
        # Runs on ALL ranks. ``self.enabled`` is config-driven and never flipped at
        # runtime, so every rank agrees on whether the barrier in
        # ``on_validation_epoch_end`` fires -- this keeps the collective in step.
        if not self.enabled:
            return

        self.rank = int(getattr(trainer, "rank", 0))
        self.world_size = int(getattr(trainer, "world_size", 1))

        # Create the gloo (CPU) barrier group. Collective -> all ranks must call it,
        # so this happens before the rank-0-only geometry build below. The barrier
        # is held for the *entire* rank-0 measurement, which on a cold page cache can
        # take far longer than gloo's 30-min default (the first epoch reads ~140
        # tiny gzip NIfTI frames per patient). Give it the same generous headroom as
        # the NCCL group (DDP_TIMEOUT_MIN, default 120m) so it doesn't abort rank 1.
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            timeout_min = int(os.environ.get("DDP_TIMEOUT_MIN", "120"))
            try:
                self._gather_group = dist.new_group(
                    backend="gloo", timeout=timedelta(minutes=timeout_min)
                )
            except Exception as exc:  # pragma: no cover - falls back to NCCL barrier
                logger.warning(
                    f"FlowValidation: could not create gloo barrier group ({exc}); "
                    "falling back to the default (NCCL) group."
                )
                self._gather_group = None

        # Only rank 0 owns the geometry / measurement.
        if not trainer.is_main_process:
            return

        patient_ids = self._select_patient_ids(trainer)
        if not patient_ids:
            logger.warning(
                "FlowValidation: no patients selected; epochs will no-op (other "
                "ranks still barrier so DDP stays in step)."
            )
            return

        path_config = load_path_config(self.config_name)
        for pid in patient_ids:
            try:
                patient = Patient(
                    path_config=path_config,
                    phonetic_id=pid,
                    config=self.config_name,
                )
            except Exception as exc:
                logger.warning(f"FlowValidation: cannot build Patient {pid}: {exc}")
                continue
            cache_path = build_downsampled_cache(
                patient, self.downsampled_folder, seg_threshold=self.seg_threshold
            )
            if cache_path is None:
                logger.warning(
                    f"FlowValidation: degenerate/missing auto-flow geometry for {pid}; "
                    "skipping."
                )
                continue
            self._patients[pid] = patient
            self._geometry[pid] = PatientFlowGeometry(cache_path)
            try:
                self._venc[pid] = float(patient.venc)
            except Exception as exc:
                logger.warning(f"FlowValidation: cannot read VENC for {pid}: {exc}; skipping")
                self._patients.pop(pid, None)
                self._geometry.pop(pid, None)

        logger.info(
            f"FlowValidation ready for {len(self._geometry)} patients: "
            f"{', '.join(self._geometry) or '(none)'}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _select_patient_ids(self, trainer: "BaseTrainer") -> List[str]:
        if self.explicit_ids:
            return self.explicit_ids
        if trainer.val_dataset is None:
            return []
        seen: List[str] = []
        for subject in trainer.val_dataset.dry_iter():
            pid = getattr(subject, "patient_id", None)
            if pid and pid not in seen:
                seen.append(pid)
            if 0 <= self.num_patients <= len(seen):
                break
        return seen

    # ------------------------------------------------------------------
    # Per-epoch measurement
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        # Runs on ALL ranks. The cadence gate below is config/epoch-driven, so it
        # is identical across ranks -- they either all run (rank 0 measures, the
        # rest barrier) or all skip, keeping the barrier collective in lock-step.
        if not self.enabled:
            return

        is_last_epoch = epoch == trainer.cfg.train.num_epochs - 1
        first = epoch == 0 and self.run_on_first_epoch
        if not (first or is_last_epoch or epoch % self.every_n_epochs == 0):
            return
        if "generator" not in trainer.models:
            return

        try:
            if trainer.is_main_process:
                self._measure_all(trainer, epoch)
        finally:
            # Keep the other ranks parked while rank 0 measures. A collective
            # barrier here busy-spins (gloo and NCCL both poll), and empirically
            # even one spinning rank drags rank 0's measurement ~10-20x slower
            # despite 23 idle cores. So the other ranks *sleep* (time.sleep, which
            # truly deschedules them) until rank 0 signals completion via a sentinel
            # file, then everyone meets at one short barrier to resynchronise.
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                self._rendezvous(epoch)

    def _rendezvous(self, epoch: int) -> None:
        sentinel = self.output_dir / f".flow_epoch_{epoch}_done"
        timeout_s = int(os.environ.get("DDP_TIMEOUT_MIN", "120")) * 60
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(str(time.time()))
            dist.barrier(group=self._gather_group)
            try:
                sentinel.unlink()
            except OSError:
                pass
        else:
            start = time.time()
            while not sentinel.exists():
                if time.time() - start > timeout_s:
                    logger.warning(
                        "FlowValidation: rank %d timed out waiting for rank 0 "
                        "measurement (%ds); proceeding to barrier.",
                        self.rank, timeout_s,
                    )
                    break
                time.sleep(1.0)
            dist.barrier(group=self._gather_group)

    def _measure_all(self, trainer: "BaseTrainer", epoch: int) -> None:
        generator = trainer.models["generator"]
        # Use the underlying module for the forward pass to avoid any DDP-wrapper
        # collectives (other ranks are parked at the barrier, not in forward).
        model_fwd = generator.module if hasattr(generator, "module") else generator
        was_training = generator.training
        generator.eval()
        device = trainer.device

        rows: List[Dict[str, Any]] = []
        t_start = time.perf_counter()
        n_measured = 0
        for pid, geometry in self._geometry.items():
            patient = self._patients[pid]
            try:
                per_patient = self._measure_patient(patient, geometry, model_fwd, device)
            except Exception as exc:
                logger.warning(f"FlowValidation: measurement failed for {pid}: {exc}")
                continue
            n_measured += 1
            for variant, res in per_patient.items():
                rows.append(
                    {
                        "epoch": epoch,
                        "global_step": trainer.global_step,
                        "patient_id": pid,
                        "variant": variant,
                        "Ao": res["Ao"],
                        "PA": res["PA"],
                        "Qp_Qs": res["Qp_Qs"],
                    }
                )

        if was_training:
            generator.train()

        elapsed = time.perf_counter() - t_start
        if not rows:
            logger.warning("FlowValidation: produced no measurements this epoch.")
            return

        per_patient = elapsed / n_measured if n_measured else float("nan")
        logger.info(
            f"FlowValidation epoch {epoch}: measured {n_measured} patients in "
            f"{elapsed:.1f}s ({per_patient:.1f}s/patient)"
        )
        self._log(trainer, epoch, rows, elapsed)

    def _measure_patient(
        self,
        patient: Patient,
        geometry: PatientFlowGeometry,
        generator: torch.nn.Module,
        device: torch.device,
    ) -> Dict[str, Dict[str, float]]:
        pid = patient.identifier
        n_t = geometry.n_timepoints
        venc = self._venc[pid]

        # Uncorrected is needed every epoch (it's the base the model adds to and the
        # velocity input channels), but its *flow* is static -- measure it once.
        # Optionally keep the array in RAM to skip re-reading it each event.
        _t = time.perf_counter()
        _dt: Dict[str, float] = {}
        uncorrected = self._uncorrected_cache.get(pid)
        if uncorrected is None:
            uncorrected = load_downsampled_velocity(
                patient, self.downsampled_folder, n_t, corrected=False
            )  # (X, Y, Z, T, 3) mm/s
            if self.cache_uncorrected_in_ram:
                self._uncorrected_cache[pid] = uncorrected
        _dt["load_unc"] = time.perf_counter() - _t; _t = time.perf_counter()
        if pid not in self._baseline:
            gt = load_downsampled_velocity(
                patient, self.downsampled_folder, n_t, corrected=True
            )
            _dt["load_gt"] = time.perf_counter() - _t; _t = time.perf_counter()
            self._baseline[pid] = {
                "uncorrected": geometry.measure(uncorrected),
                "gt": geometry.measure(gt),
            }
            del gt
            _dt["measure_base"] = time.perf_counter() - _t; _t = time.perf_counter()

        model = self._model_corrected(patient, n_t, uncorrected, venc, generator, device)
        _dt["model"] = time.perf_counter() - _t; _t = time.perf_counter()
        result = dict(self._baseline[pid])
        result["model"] = geometry.measure(model)
        _dt["measure_model"] = time.perf_counter() - _t
        logger.info(
            "FlowValidation[rank %d][%s] timing: %s",
            self.rank,
            pid,
            ", ".join(f"{k}={v:.1f}s" for k, v in _dt.items()),
        )
        return result

    def _model_corrected(
        self,
        patient: Patient,
        n_t: int,
        uncorrected: np.ndarray,
        venc: float,
        generator: torch.nn.Module,
        device: torch.device,
    ) -> np.ndarray:
        """Run the generator over every timepoint -> model-corrected field (mm/s).

        ``model_corrected = uncorrected + pred_correction * venc``, where
        ``pred_correction`` is the VENC-normalised correction head output. Inputs
        are assembled by indexing the magnitude frames (loaded once, normalised
        like the training transform) and reusing the already-in-memory
        ``uncorrected`` field for the velocity channels -- no per-timepoint subject
        rebuild or extra disk reads.
        """
        pid = patient.identifier
        mags_stacked = self._mag_cache.get(pid)
        if mags_stacked is None:
            frames = [_rescale01(a) for a in load_downsampled_mag_frames(
                patient, self.downsampled_folder, n_t
            )]
            # Cache the magnitudes already stacked into one contiguous (T,X,Y,Z)
            # float32 array. np.stack rebuilds (and copies ~80 MB) on every call,
            # and that CPU copy is exactly what balloons under whole-box memory
            # pressure; caching the stacked form makes later epochs a zero-copy
            # view + a pure PCIe upload.
            mags_stacked = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)
            if self.cache_uncorrected_in_ram:
                self._mag_cache[pid] = mags_stacked

        # All array assembly happens on the GPU. Doing it in numpy on the CPU was
        # the dominant cost: building the (T, C, X, Y, Z) input and the output
        # field moves ~0.6 GB/patient through host RAM, which is memory-bandwidth
        # bound and degrades severely under whole-box memory pressure (other ranks,
        # dataloader workers, page cache). On the GPU it is HBM-bound and tiny. The
        # only host<->device traffic is uploading the static inputs once and
        # bringing the final corrected field back for the (numpy/scipy) geometry
        # measure. The generator forward runs in chunks of ``tp_chunk`` timepoints.
        _sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        _b = time.perf_counter()
        n_mag = len(self._mag_offsets)
        mags_t = torch.from_numpy(mags_stacked).to(device)  # (T, X, Y, Z)
        unc_t = torch.from_numpy(uncorrected).to(device)  # (X,Y,Z,T,3), zero-copy view
        vel_norm = torch.clamp(unc_t / venc, -1.0, 1.0)  # (X, Y, Z, T, 3)
        out_t = torch.empty_like(unc_t)
        _sync(); t_up = time.perf_counter() - _b

        t_fwd = 0.0
        for s in range(0, n_t, self.tp_chunk):
            e = min(s + self.tp_chunk, n_t)
            chans = []
            for t in range(s, e):
                per_t = [mags_t[(t + off) % n_t] for off in self._mag_offsets]
                per_t += [vel_norm[:, :, :, t, c] for c in range(3)]
                chans.append(torch.stack(per_t, dim=0))  # (C, X, Y, Z)
            chunk = torch.stack(chans, dim=0)  # (bs, C, X, Y, Z)
            _bf = time.perf_counter()
            with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
                pred = generator(chunk)
            corr = pred[:, 1:4].float()  # (bs, 3, X, Y, Z), VENC units
            for i, t in enumerate(range(s, e)):
                for c in range(3):
                    out_t[:, :, :, t, c] = unc_t[:, :, :, t, c] + corr[i, c] * venc
            _sync(); t_fwd += time.perf_counter() - _bf

        _b = time.perf_counter()
        out = out_t.cpu().numpy()
        t_down = time.perf_counter() - _b
        logger.info(
            "FlowValidation[rank %d][%s] model breakdown: up=%.1fs fwd+asm=%.1fs down=%.1fs",
            self.rank, patient.identifier, t_up, t_fwd, t_down,
        )
        return out

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(
        self,
        trainer: "BaseTrainer",
        epoch: int,
        rows: List[Dict[str, Any]],
        elapsed: Optional[float] = None,
    ) -> None:
        if self.write_csv:
            csv_path = self.output_dir / "flow_metrics.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["epoch", "global_step", "patient_id", "variant", "Ao", "PA", "Qp_Qs"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)

        # Aggregate across patients per variant. Geometry-broken patients are kept
        # in the per-patient CSV above but dropped from the cohort aggregates.
        excluded = set(self.exclude_ids)
        agg: Dict[str, float] = {}
        by_variant = {
            v: [r for r in rows if r["variant"] == v and r["patient_id"] not in excluded]
            for v in _VARIANTS
        }
        for variant, vrows in by_variant.items():
            if not vrows:
                continue
            for key in ("Ao", "PA", "Qp_Qs"):
                vals = [r[key] for r in vrows if np.isfinite(r[key])]
                if vals:
                    agg[f"flow/{variant}_{key}_mean"] = float(np.mean(vals))

        # Error metrics vs the GT-corrected reference. For each key we report:
        #   *_model_vs_gt_mae        mean |model - gt|              (flow units, L/min)
        #   *_uncorrected_vs_gt_mae  mean |uncorr - gt|             (baseline, flow units)
        #   *_model_vs_gt_nmae       sum|model-gt| / sum|gt|        (unitless: error as a
        #                            fraction of true flow magnitude)
        #   *_residual_fraction      sum|model-gt| / sum|uncorr-gt| (1 = no better than
        #                            uncorrected, 0 = perfect). Ratio-of-sums avoids the
        #                            per-patient blow-up when uncorr already ~= gt.
        #   *_error_reduction        1 - residual_fraction          (fraction of the
        #                            correctable error the model actually removed)
        gt_by_pid = {r["patient_id"]: r for r in by_variant["gt"]}
        unc_by_pid = {r["patient_id"]: r for r in by_variant["uncorrected"]}
        for key in ("Ao", "PA", "Qp_Qs"):
            m_errs: List[float] = []
            u_errs: List[float] = []
            gt_abs: List[float] = []
            for r in by_variant["model"]:
                pid = r["patient_id"]
                gtr = gt_by_pid.get(pid)
                if gtr is None:
                    continue
                g, m = gtr[key], r[key]
                if not (np.isfinite(g) and np.isfinite(m)):
                    continue
                m_errs.append(abs(m - g))
                gt_abs.append(abs(g))
                ur = unc_by_pid.get(pid)
                u = ur[key] if ur is not None else float("nan")
                u_errs.append(abs(u - g) if np.isfinite(u) else float("nan"))
            if not m_errs:
                continue
            m_arr = np.asarray(m_errs)
            gt_arr = np.asarray(gt_abs)
            u_arr = np.asarray(u_errs)
            agg[f"flow/{key}_model_vs_gt_mae"] = float(m_arr.mean())
            if gt_arr.sum() > 1e-8:
                agg[f"flow/{key}_model_vs_gt_nmae"] = float(m_arr.sum() / gt_arr.sum())
            valid_u = np.isfinite(u_arr)
            if valid_u.any():
                agg[f"flow/{key}_uncorrected_vs_gt_mae"] = float(u_arr[valid_u].mean())
                denom = float(u_arr[valid_u].sum())
                if denom > 1e-8:
                    resid = float(m_arr[valid_u].sum() / denom)
                    agg[f"flow/{key}_residual_fraction"] = resid
                    agg[f"flow/{key}_error_reduction"] = 1.0 - resid

        if elapsed is not None:
            agg["flow/measure_seconds"] = float(elapsed)

        logger.info(
            f"FlowValidation epoch {epoch}: "
            + ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(agg.items()))
        )

        if self.log_to_wandb and WANDB_AVAILABLE:
            try:
                wandb.log(agg, step=trainer.global_step)
            except Exception as exc:
                logger.warning(f"FlowValidation: wandb.log failed: {exc}")
