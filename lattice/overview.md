---
description: >-
  What capture_lattice.py does, what it deliberately does not do, and how the
  files it writes reach Chloros.
---

# Overview

`capture_lattice.py` controls MAPIR **LATTICE** cameras (M3C color / M3M mono)
from your own flight computer and writes raw frames to Chloros-compatible
TIFFs. LATTICE cameras are GigE Vision machine-vision cameras, driven through
the **Arena SDK** and its `arena_api` Python wrapper.

The script:

* configures each camera for raw capture — Bayer mosaic (M3C) or mono (M3M),
  on-camera ISP off;
* hardware-syncs a multi-camera array over the MAPIR **M8 sync cables** (one
  master drives `ExposureActive` on Line2; the slaves trigger off that edge);
* saves each frame as a raw TIFF stamped with **serial + model + exposure**, so
  Chloros detects it, groups the synced shot, and fetches the camera's factory
  calibration by serial at import.

## What it does not do

{% hint style="warning" %}
It does **not** debayer, calibrate, or apply any index. Frames land on disk
exactly as the sensor produced them. Capture is yours; Chloros does the science
at import.
{% endhint %}

The one thing you must get right is **identity**. The camera serial is the
calibration fetch key — Chloros pulls that exact camera's factory calibration
from the cloud by serial. The script reads it off the camera (`DeviceSerialNumber`),
along with the model string from `DeviceUserID`, so get the camera powered and
discoverable before capturing.

## Where the pieces live

| File | Role |
| --- | --- |
| `capture_lattice.py` | Camera discovery, raw configuration, sync wiring, capture loop |
| `mapir_metadata.py` | `write_lattice_raw_tiff()` and `lattice_capture_filename()` — the Chloros ingest contract |
| `selftest.py` | Checks the written output against what Chloros reads on import |

All three are in
[`mapircamera/chloros_scripts`](https://github.com/mapircamera/chloros_scripts).

## Next steps

{% content-ref url="host-and-arena-sdk.md" %}
[host-and-arena-sdk.md](host-and-arena-sdk.md)
{% endcontent-ref %}

{% content-ref url="capturing-raw-frames.md" %}
[capturing-raw-frames.md](capturing-raw-frames.md)
{% endcontent-ref %}
