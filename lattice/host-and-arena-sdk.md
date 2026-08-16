---
description: >-
  Installing the Arena SDK and arena_api, host hardware requirements, and
  network setup for LATTICE capture.
---

# Host & Arena SDK Setup

## Python dependencies

```bash
python -m pip install -r requirements.txt
```

That covers `numpy` and `Pillow` (which `mapir_metadata.py` uses to write the
raw TIFF and its EXIF). The camera driver is separate.

## Arena SDK

`capture_lattice.py` talks to the cameras through the **Arena SDK** and its
`arena_api` Python wrapper.

{% hint style="warning" %}
`arena_api` is **not** installable from PyPI on its own. Install the native
Arena SDK for your platform first (Windows, and Linux x86-64 / arm64 including
Jetson and Raspberry Pi builds), then its Python package.
{% endhint %}

It is intentionally left out of `requirements.txt` so that
`pip install -r requirements.txt` still works for the DAQ-only use case. If it
is missing, the script exits with an explicit message rather than a traceback.

`record_daq.py` does **not** need the Arena SDK.

## Host hardware

These scripts do no image processing — they just move raw data to disk — so CPU
load is low. What matters is the network interface, a bit of RAM, and write
speed. The Arena SDK sets a hard floor: it ships only for **64-bit ARM
(aarch64) and x86-64** (no 32-bit / ARMv6 / ARMv7 build), and the cameras are
wired Gigabit Ethernet.

| | Recommendation |
| --- | --- |
| **Minimum** | A **64-bit OS** on 64-bit ARM or x86-64, Gigabit Ethernet (onboard, or a USB-to-Gigabit adapter — use a USB 3 port, since USB 2.0 caps around 480 Mbps), ~1 GB free RAM for one camera (budget ≈70 MB more per additional camera for frame buffers). Board floor: **Raspberry Pi 4 (2 GB+) on 64-bit Raspberry Pi OS / Ubuntu.** A microSD card is fine for a single camera at low frame rates. |
| **Ideal** | **NVIDIA Jetson Orin Nano / NX / AGX** (or an x86-64 mini-PC), 4 GB+ RAM, with an **SSD** (USB 3 / NVMe). Arrays need the SSD: frames are uncompressed (~6.3 MB each), so e.g. 5 cameras at 2 fps is ~60 MB/s of sustained writes a microSD card cannot keep up with. |

{% hint style="danger" %}
**A Raspberry Pi Zero cannot run the cameras** — it is ARMv6, which the Arena
SDK does not support. Adding a USB-Gigabit adapter does not get around that.
The smallest practical camera host is a Raspberry Pi 4 on a 64-bit OS.
{% endhint %}

Storage and network bandwidth both scale with camera count × frame rate, so
step up to a Jetson or x86-64 host for larger arrays.

## Network

Put the host NIC **on the cameras' subnet**, and enable **jumbo frames** if your
switch supports them. The capture script asks the SDK to negotiate the largest
safe GigE packet size and enables lost-packet resends, but it cannot fix a host
that is not on the right subnet — the symptom is simply that no cameras are
found:

```
No cameras found. Check power, cabling, and that the host NIC is on the cameras' subnet.
```
