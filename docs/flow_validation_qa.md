# Flow Validation Geometry QA

Manual review of the precomputed auto-flow geometry (splines + segmentations)
used by the in-loop flow validation metric on the **validation split**.

## Why most geometry errors are tolerable

The flow metric measures the **same voxel set** across all three velocity
fields per patient:

- uncorrected
- ground-truth corrected
- CNN-corrected (model output)

Because the comparison is **relative** (does the model recover the GT-corrected
flow?), segmentation/spline placement errors are *common-mode* and cancel out.
They would only bias **absolute** clinical flow values (Ao, PA, Qp:Qs), not the
relative model-improvement signal. The notes below are flagged for awareness,
not exclusion.

## Status summary

- 48 validation patients total
- **36 usable** (passed the automated degenerate-localization guard)
- **12 excluded** (degenerate / failed geometry — see bottom)

## Per-patient QA notes (36 usable)

Reviewed visually via the aortic/pulmonary spline + segmentation GIFs.

| Patient | Note |
|---|---|
| Curintod | Segmentations are the correct vessel but a bit messy. |
| Dunurul | **Vessels swapped**: "aortic" segmentation is actually the PA; "pulmonary" segmentation is actually the aorta. |
| Farare | Both splines look incorrect. |
| Fephowi | Artificial valve present → both segmentations land on the same vessel. |

The flagged patients above are retained because the errors are common-mode for
the relative metric. All other usable patients (32) were reviewed and look good:

Alernscet, Beborep, Ciphiscap, Enogar, Fajupo, Goquogi, Gotedol, Gririeyoze,
Heyatam, Incepey, Koclabup, Lamiemmos, Leerscoopub, Lucathi, Mibago, Nilirnay,
Niritha, Orupom, Oysertes, Peernrinkjey, Quanagug, Quonirab, Shidunu, Sleyolof,
Sotomu, Suquidu, Tijune, Turndernize, Utusin, Vahapug, Vekeeyu, Voufafu

## Excluded (12) — degenerate / failed geometry

Caught by the `(0,0,0)` missed-landmark guard (`localization_is_valid`). All 12
generated both spline CSVs, but LocNet missed 2-5 intermediate landmarks each:

- **Hard failures (5 missed → crash / zero-length tangent)**: Begaca, Bojiho,
  Cirnurey, Darabos, Ipunkan, Mabaydem, Scibebum, Totekan
- **Silent corruption (2-4 missed → built a corrupt cache, now removed)**:
  Cakoohu, Gukietet, Miguton, Stomurquos

### Root cause

Single shared cause: auto-flow's **LocNet detects the distal endpoints (AV,
Full Ao, Full PA) but misses the proximal/mid intermediate landmarks**, which
fall back to voxel `(0,0,0)`. Frequency across the 12:

| Missed landmark | count |
|---|---|
| Proximal AAo | 12 |
| Proximal MPA | 11 |
| PV | 10 |
| MPA | 9 |
| Mid AAo | 8 |
| Full PA with branches | 2 |
| AV | 1 |

This is a detection limitation in auto-flow's LocNet, not a bug in this repo's
flow pipeline or in spline generation. Recovering these patients would require
improving/retraining LocNet or a manual landmark-correction pass; deferred.
