"""Flow-rate math (numpy only), matching auto-flow's ``compute_flow``.

Given a through-plane velocity volume ``(R, C, n_planes, T)`` (mm/s) and a
matching segmentation ``(R, C, n_planes, T)`` (soft mask in [0, 1]), integrate to
a per-plane volumetric flow rate in L/min.

Pipeline (identical to auto-flow):
  flow(R,C,plane,t) = v_through * conversion * pixel_area * seg
  per-plane, per-time instantaneous flow (L/s) = sum over R, C
  stroke volume (L/beat) = trapezoidal integral over one cardiac cycle
  volumetric flow rate (L/min) = stroke_volume * bpm
"""

from __future__ import annotations

import numpy as np

# mm^3/s -> L/s: (0.1 cm/mm)^3 converts mm^3 -> cm^3, then /1000 cm^3 -> L.
CONVERSION_FACTOR = (0.1) ** 3 * 1.0 / 1000.0


def instantaneous_flow(
    through_plane: np.ndarray,
    seg: np.ndarray,
    *,
    conversion_factor: float = CONVERSION_FACTOR,
    pixel_area: float = 1.0,
) -> np.ndarray:
    """Per-plane, per-time instantaneous flow (L/s); shape ``(n_planes, T)``.

    Mirrors auto-flow ``calculate_flow``: scale, mask, sum over the in-plane axes.
    """
    flow = np.asarray(through_plane, float) * conversion_factor * pixel_area
    flow = flow * np.asarray(seg, float)
    return flow.sum(axis=(0, 1))  # -> (n_planes, T)


def volumetric_flow_rate(flow_per_time: np.ndarray, bpm: float) -> np.ndarray:
    """Integrate instantaneous flow over one cycle -> L/min, per plane.

    ``flow_per_time`` is ``(n_planes, T)`` in L/s. Uses the same trapezoidal rule
    over ``T`` samples spanning one beat (dT = (60/bpm)/T) as auto-flow.
    """
    flow_per_time = np.asarray(flow_per_time, float)
    n_t = flow_per_time.shape[-1]
    sec_per_beat = 60.0 / bpm
    dt = sec_per_beat / n_t
    # Trapezoid over consecutive samples (no wrap), matching auto-flow exactly.
    stroke_volume = np.trapz(flow_per_time, dx=dt, axis=-1)
    # np.trapz integrates t[0..T-1]; auto-flow sums dT*(f[i+1]+f[i])/2 over i=0..T-2,
    # which is identical to np.trapz with uniform dx.
    return stroke_volume * bpm  # L/min per plane


def measure_flow(
    through_plane: np.ndarray,
    seg: np.ndarray,
    bpm: float,
    *,
    conversion_factor: float = CONVERSION_FACTOR,
    pixel_area: float = 1.0,
) -> dict:
    """Full reduction: per-plane L/min plus mean/std across planes.

    Returns ``{"per_plane": (n_planes,), "mean": float, "std": float}``.
    """
    inst = instantaneous_flow(
        through_plane, seg, conversion_factor=conversion_factor, pixel_area=pixel_area
    )
    per_plane = volumetric_flow_rate(inst, bpm)
    return {
        "per_plane": per_plane,
        "mean": float(np.mean(per_plane)),
        "std": float(np.std(per_plane)),
    }
