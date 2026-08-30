---
description: >-
  The three DAQ transports, the scripts that speak them, and which data channel
  to use for what.
---

# Overview

MAPIR **DAQ** light sensors measure downwelling spectral irradiance. Recording
one during a flight is what turns LATTICE radiance into **percent reflectance**
at import — with no DAQ you still get radiance, but not reflectance.

These scripts speak the DAQ wire protocol directly, with **no MAPIR SDK**, over
all three transports:

| Device | Transport | Python dependency |
| --- | --- | --- |
| **DAQ-U** | USB serial | `pyserial` |
| **DAQ-M** | Bluetooth LE (Nordic UART Service) | `bleak` |
| **DAQ-E** | Ethernet (JSON control + raw TCP + multicast) | none — standard library |


## The scripts

| Script | Purpose |
| --- | --- |
| [`record_daq.py`](recording-a-flight.md) | Record raw spectra from a DAQ-U / DAQ-M / DAQ-E to a Chloros-compatible `.daq` |
| [`daq_stream.py`](multi-sensor-streaming.md) | Listen to any number of DAQ-E sensors over multicast, raw or calibrated, and record what arrives |
| [`daq_cal.py`](offline-calibration.md) | Read or write a DAQ-E's onboard calibration and [cap profile](caps.md) — no cloud, no Chloros |
| [`mapir_metadata.py`](daq-file-format.md) | `DaqWriter` — the `.daq` SQLite format Chloros imports |

## Install

```bash
python -m pip install -r requirements.txt
```

* **DAQ-U** needs `pyserial`. On Linux add yourself to the `dialout` group for
  serial access: `sudo usermod -aG dialout $USER` (then re-login).
* **DAQ-M** needs `bleak`. On Linux it uses BlueZ (`sudo apt install bluez`);
  Jetson and Raspberry Pi work out of the box.
* **DAQ-E** needs nothing beyond the standard library.

DAQ recording has a tiny footprint — a few hundred small readings per second,
parsed and written to SQLite. A **Raspberry Pi Zero 2 W** handles it; any Pi 4/5
or Jetson is far more than enough. Python 3.8+ and ~256 MB free RAM is the floor.

## DAQ-E data channels

A DAQ-E on firmware **1.7.0+** emits two spectral streams on separate multicast
groups: **raw** counts (always) and **calibrated** W/m²/nm (once the device
carries coefficients). Older firmware emits raw only.

| Channel | Wire | Content | Script |
| --- | --- | --- | --- |
| Raw, unicast | TCP `5000` | Raw counts, **one client at a time** | `record_daq.py` |
| Raw, multicast | UDP `239.10.10.10:5002` | Raw counts, any number of listeners | `daq_stream.py` |
| Calibrated, multicast | UDP `239.10.10.11:5003` | W/m²/nm, when the device carries coefficients | `daq_stream.py --calibrated` |
| Control | TCP `5001` | JSON: config, status, bundle / profile / cert | `daq_cal.py` |

{% hint style="info" %}
**Raw is always the reprocessable one.** It is the sensor's firmware output byte
for byte, so a recording made from it can be re-calibrated later against a
revised bundle. Prefer it for anything you intend to keep.
{% endhint %}

Every frame says which stream it came from, from the **frame's own flag bit**
rather than from the group the script joined — so a recording describes what
actually arrived. See [Multi-Sensor Streaming](multi-sensor-streaming.md).

## No camera? Still a project

Importing a `.daq` writes the calibrated spectra back out as a `.daq` **and** a
`.csv` of spectral irradiance, for DAQ-U, DAQ-M and DAQ-E alike, with no imagery
involved.

{% content-ref url="light-sensor-only.md" %}
[light-sensor-only.md](light-sensor-only.md)
{% endcontent-ref %}

## The wire protocol

All three transports share one NSP32-style framing:

* **Command:** `03 BB <cmd> <user> [payload…] <checksum>`
* **Response:** same shape
* `checksum = ((~sum(bytes_before_checksum)) + 1) & 0xFF` — a valid packet has
  `sum(whole_packet) & 0xFF == 0`
* All multi-byte fields little-endian

Response length is fixed per command code, which is how the framer knows how
much to read after the `03 BB` prefix. For a 135-point sensor, `GetWavelength`
is 279 bytes and `GetSpectrum` is 565.
