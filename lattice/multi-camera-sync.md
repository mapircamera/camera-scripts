---
description: >-
  How --sync cable drives a LATTICE array off one trigger line over the M8 sync
  harness, and the firmware ordering quirks that make it work.
---

# Multi-Camera Hardware Sync

With `--sync cable` (the default), the **master** camera is software-triggered
and drives an `ExposureActive` pulse out on **Line2**. Every slave triggers off
that rising edge over the **MAPIR M8 sync cable** (pin 2 → Line2). That gives
sub-frame, simultaneous exposure across the array with no PTP — which is what a
single cabled rig needs.

```bash
python capture_lattice.py --sync cable --master 213602328 --interval 1.0
```

If you do not pass `--master`, the camera with the **lowest serial** is used and
the script says so.

{% hint style="info" %}
Syncing cameras that are *not* cabled together, or measuring the exact skew
between them, is what PTP is for. That is not implemented in these scripts.
{% endhint %}

## Software mode

`--sync software` triggers each camera individually with no GPIO. It is the
single-camera / no-cable path.

{% hint style="danger" %}
Multiple cameras in `software` mode are **not** hardware-synchronized. The
script prints a warning if you do it anyway. Use `--sync cable` with M8 sync
cables for simultaneous exposure.
{% endhint %}

## The GenICam wiring

**Master:**

| Node | Value |
| --- | --- |
| `TriggerMode` | `On` |
| `TriggerSource` | `Software` |
| `LineSelector` | `Line2` |
| `LineMode` | `Output` |
| `LineSource` | `ExposureActive` |

**Slave:**

| Node | Value |
| --- | --- |
| `TriggerMode` | `Off`, then `On` last |
| `LineSelector` | `Line2` |
| `LineMode` | `Input` |
| `LineSource` | `ExposureActive` |
| `TriggerSource` | `Line2` |
| `TriggerActivation` | `RisingEdge` |
| `TriggerOverlap` | `PreviousFrame` |

## Ordering quirks

The slave sequence above is not arbitrary. Three behaviors observed on camera
firmware dictate it:

1. **`TriggerMode` must be `Off` while the trigger wiring is rewritten.** Some
   firmware locks the configuration while armed.
2. **Force `LineMode = Input` *before* setting `TriggerSource`.** The
   bidirectional line's mode persists in flash; if it was ever left as `Output`,
   the firmware refuses to bind `TriggerSource` to it.
3. **A slave with `LineSource = Off` goes deaf to the cable pulse**, even with
   `LineMode = Input`. The input sensing path has to be biased with a non-`Off`
   `LineSource`, so it is set to `ExposureActive` explicitly.

{% hint style="warning" %}
A slave that never fires is almost always quirk 2 or 3 — a line left in
`Output` mode by earlier software, or a `LineSource` of `Off`. Re-running the
script fixes both, since it rewrites the full sequence every time.
{% endhint %}

## What the array produces

Every camera in one trigger writes a frame with the **same sequence number** in
its filename. Chloros groups a multi-camera capture on the part of the filename
after the serial, so that shared sequence number is what makes the shot a single
grouped capture rather than N unrelated frames.

{% content-ref url="tiff-output-and-metadata.md" %}
[tiff-output-and-metadata.md](tiff-output-and-metadata.md)
{% endcontent-ref %}
