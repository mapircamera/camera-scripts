---
description: >-
  Running record_daq.py against a DAQ-U, DAQ-M, or DAQ-E to produce a
  Chloros-compatible .daq recording.
---

# Recording a Flight

`record_daq.py` records the sensor's **raw** firmware-output spectrum to a
`.daq` file and stamps it so Chloros fetches that sensor's factory calibration
from the cloud **by serial** and applies it at import.

## Usage

```bash
# DAQ-U over USB serial
python record_daq.py u --port COM7              # Windows
python record_daq.py u --port /dev/ttyUSB0      # Linux / Jetson / Pi
```

```bash
# DAQ-M over Bluetooth LE (the sensor's BLE address)
python record_daq.py m --mac AA:BB:CC:DD:EE:FF
```

```bash
# DAQ-E over Ethernet (the sensor's IP)
python record_daq.py e --host 192.168.1.50
```

Press **Ctrl-C** to stop. The script records continuously; mount the sensor
upward-facing (downwelling) and run it for the whole flight.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--integration-time MS` | Integration time per reading | `32` |
| `--frame-avg N` | Frames averaged per reading | `3` |
| `--no-ae` | Disable auto-exposure (use the fixed integration time) | AE on |
| `--frames N` | Stop after N readings | until `Ctrl-C` |
| `--duration S` | Stop after S seconds | until `Ctrl-C` |
| `--output PATH` | Output `.daq` path | `<kind>_<timestamp>.daq` |
| `--csv [PATH]` | Also write a CSV of what this run records | off |
| `--device-name NAME` | Free-text label stored in the file | empty |
| `--cap-id ID` | Which cap is fitted — see [Caps](caps.md) | the cap this model ships wearing |

Bare `--csv` puts the file beside the `.daq`; give a `PATH` to place it. Without
`--calibrate` it holds the same **raw counts** that go into the `.daq`, which is
the only CSV a DAQ-U or DAQ-M can produce — neither carries a bundle to
calibrate against locally.

Transport-specific:

| Option | Applies to | Meaning | Default |
| --- | --- | --- | --- |
| `--port` | DAQ-U | Serial port | required |
| `--mac` | DAQ-M | BLE address | required |
| `--host` | DAQ-E | IP address | required |
| `--raw-port` | DAQ-E | Raw spectral TCP port | `5000` |
| `--control-port` | DAQ-E | JSON control TCP port | `5001` |
| `--token` | DAQ-E | Control-channel auth token | empty |

`--calibrate` and `--require-profiles` are DAQ-E only and covered in
[Offline Calibration](offline-calibration.md). `--cap-id` applies to every model
— on a raw recording it *declares* what is fitted rather than applying anything,
which is what [Caps](caps.md) is about.

{% hint style="warning" %}
A DAQ-E with a token provisioned (done at the calibration station) refuses every
command except `hello` without `--token`. The symptom is `auth_failed` on
connect or on `--calibrate`'s `get_calibration`. Passing `--token` is harmless
on unprovisioned units.
{% endhint %}

## What happens on connect

```
Connecting to daq-e ...
  serial (cal key): 11-22-33-44-55   fw: 1.7.0
  cap declared:     sunshine_cosine  (assumed -- fleet default; pass --cap-id if wrong)
Recording RAW to: /data/daq-e_20260816_142233.daq
  (Chloros will fetch this sensor's calibration by serial at import)
  Ctrl-C to stop.
```

The cap line is printed on **every** run, and the file records whether it was
assumed or stated. Getting it wrong is a 20–30× error that nothing downstream
can detect, so it is worth a glance:

{% content-ref url="caps.md" %}
[caps.md](caps.md)
{% endcontent-ref %}

The serial number is read from the device itself:

* **DAQ-U / DAQ-M** — from the `GetSensorId` command, formatted `XX-XX-XX-XX-XX`.
* **DAQ-E** — from the JSON control channel's `hello` response (`name`), which
  also reports firmware.

The wavelength grid is queried once as a liveness check but **not** stored —
Chloros gets the grid from the calibration bundle at import.

{% hint style="danger" %}
Get the device powered and discoverable so the script can read its **real**
serial. That serial is the only key Chloros has for finding this sensor's
factory calibration.
{% endhint %}

## Auto-exposure and integration time

Auto-exposure is **on** by default and moves the integration time between 1 and
500 ms. Each reading stores the integration time the sensor actually reported
for that frame, not the requested one — Chloros needs it for the
integration-aware dark model.

`--no-ae` pins the sensor to `--integration-time` instead.

## Timestamps

Each reading is stamped with **absolute** wall-clock nanoseconds
(`time.time_ns()`). Chloros matches a DAQ recording to imagery **by timestamp**,
so keep the host clock reasonably accurate for the flight.

The default output filename is stamped in **UTC**, matching the
`utc_offset_minutes = 0` declaration written into the file and the UTC EXIF
stamps on the LATTICE side.

{% content-ref url="daq-file-format.md" %}
[daq-file-format.md](daq-file-format.md)
{% endcontent-ref %}

## Getting the calibrated numbers back

The `.daq` this script writes is raw by design. Importing it into Chloros writes
the calibrated spectra back out as a `.daq` **and** a `.csv` — no camera and no
imagery required:

{% content-ref url="light-sensor-only.md" %}
[light-sensor-only.md](light-sensor-only.md)
{% endcontent-ref %}

## Stopping

A read that times out prints a warning and the loop continues. On `Ctrl-C` the
sensor is put into standby, the link is closed, and the file is committed:

```
Stopped. Wrote 4820 readings to /data/daq-e_20260816_142233.daq
```

{% hint style="info" %}
The DAQ-E's raw TCP channel takes **one client at a time**. To run several
consumers against one sensor, or several sensors at once, use the multicast
path instead:

{% content-ref url="multi-sensor-streaming.md" %}
[multi-sensor-streaming.md](multi-sensor-streaming.md)
{% endcontent-ref %}
{% endhint %}
