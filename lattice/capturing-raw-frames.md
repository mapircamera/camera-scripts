---
description: >-
  Running capture_lattice.py — command-line options, the raw camera
  configuration it applies, and how the capture loop behaves.
---

# Capturing RAW Frames

## Examples

```bash
# single camera, auto-exposure, 50 frames
python capture_lattice.py --frames 50
```

```bash
# multi-camera array, HARDWARE cable sync, pick the master by serial
python capture_lattice.py --sync cable --master 213602328 --interval 1.0
```

```bash
# fixed 5 ms exposure, run until Ctrl-C
python capture_lattice.py --exposure-us 5000
```

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--sync cable\|software` | `cable` = hardware M8 sync; `software` = single-cam / no-cable | `cable` |
| `--master SERIAL` | master camera for cable sync | lowest serial |
| `--serials A,B,…` | use only these cameras | all connected |
| `--model STR` | model to fall back to when a camera's `DeviceUserID` is empty | read from the camera |
| `--exposure-us N` | fixed exposure in microseconds | auto-exposure |
| `--frames N` | stop after N shots | until `Ctrl-C` |
| `--duration S` | stop after S seconds | until `Ctrl-C` |
| `--interval S` | wait S seconds between shots | as fast as possible |
| `--timeout-ms N` | per-frame grab timeout | `2000` |
| `--output-dir DIR` | where to write TIFFs | `.` |

`--serials` is checked against the cameras actually present; a serial you ask
for and that is not connected is an error, not a silent omission.

## Camera identification

At startup each camera reports its identity and the script prints it:

```
Found 2 LATTICE camera(s):
  213602328  LATT-M3C-L41-FRGN  (color)
  213602331  LATT-M3M-L41-F850  (mono)
```

* **Serial** comes from GenICam `DeviceSerialNumber`. This is the **calibration
  fetch key**.
* **Model** comes from `DeviceUserID` (e.g. `M3C-L41-FRGN`); the script prefixes
  `LATT-` for the EXIF `Model` tag Chloros keys on.
* Anything with `M3M` in the model is treated as **mono**; everything else as
  **color**.

{% hint style="warning" %}
If `DeviceUserID` is empty — which happens after a factory reset — the script
stops rather than guessing at the model:

```
camera 213602328: DeviceUserID is empty (was it factory reset?).
Pass the model with --model LATT-M3C-L41-FRGN.
```

`--model` supplies the missing string for that run, and the script logs which
camera it was applied to. It is a **fallback, not an override**: any camera that
still reports its own `DeviceUserID` keeps it, so a mixed array cannot be
mislabelled by one flag. Restoring the model string on the camera is still the
real fix — Chloros uses the `LATT-` prefix to detect LATTICE images at all.
{% endhint %}

## The raw configuration applied

Every camera is configured for raw scientific capture before streaming starts:

| Setting | Value | Why |
| --- | --- | --- |
| `PixelFormat` | `BayerRG12` (M3C) / `Mono12` (M3M) | Raw sensor plane, 12-bit |
| `OffsetX` / `OffsetY`, `Width` / `Height` | 0, 0, max, max | Full sensor ROI |
| `GammaEnable`, `LUTEnable`, `ColorTransformationEnable` | off | On-camera ISP off — keeps the data linear |
| `DefectCorrectionEnable` | off | On some camera firmware it halves the frame rate |
| `ExposureAuto` | `Continuous`, or `Off` + `ExposureTime` | Auto by default; fixed with `--exposure-us` |
| `AcquisitionMode` | `Continuous` | — |

Settings that a given model may not expose are attempted and skipped with a
note rather than aborting the run.

Stream-level settings: buffers are delivered in arrival order
(`StreamBufferHandlingMode = OldestFirst`), the SDK negotiates the largest safe
GigE packet size, lost-packet resends are enabled, and each camera streams with
10 buffers.

## The capture loop

Each iteration is one **shot**:

1. A single UTC timestamp is taken for the whole shot.
2. In `cable` mode the **master alone** is software-triggered and the array
   follows the wire. In `software` mode every camera is triggered individually.
3. Each camera's frame is grabbed (subject to `--timeout-ms`), its current
   exposure and gain are read, and the TIFF is written.
4. Progress prints every 10 shots.

All cameras in one shot share the same **sequence number**, which is what
Chloros uses to group a synced capture — see
[TIFF Output & Metadata](tiff-output-and-metadata.md).

A grab that fails prints a warning and is skipped; the run continues. `Ctrl-C`
stops cleanly, including during an `--interval` wait, and the script reports
what it saved:

```
Stopped. Saved 100 frames across 50 shots to /data/flight1
```

## Exposure and ISO in the file

The per-frame exposure is read back from the camera (`ExposureTime`) and stored
in EXIF in seconds. Gain is read in dB and converted to an ISO equivalent as
`100 × 10^(gain_dB / 20)`, which is the inverse of the relation Chloros uses to
recover analog gain.

{% hint style="info" %}
With auto-exposure the value read back is the current node value, so it can be
up to one frame stale. Use `--exposure-us` when you need the stamped exposure to
be exact.
{% endhint %}
