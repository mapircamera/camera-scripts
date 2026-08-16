---
description: >-
  Reading any number of DAQ-E sensors at once over UDP multicast with
  daq_stream.py, and why sensor identity in the datagram matters.
---

# Multi-Sensor Streaming

The DAQ-E publishes spectra to **UDP multicast**, so unlike the raw TCP bridge
(one client at a time, which is what `record_daq.py` uses) any number of
consumers can read the same sensor simultaneously — and any number of sensors
can share one group.

## Usage

```bash
python daq_stream.py                         # every sensor on the network
```

```bash
python daq_stream.py --calibrated            # W/m²/nm straight off the device
```

```bash
python daq_stream.py --serial 11-22-33-44-55 --csv out.csv
```

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--calibrated` | Listen to the calibrated stream instead of raw counts | raw |
| `--group` / `--port` | Override the multicast group / port | per stream |
| `--iface IP` | Local interface IP to join on | OS default |
| `--serial S` | Only show this sensor serial; repeatable | all |
| `--csv PATH` | Append every frame to a CSV | none |
| `--count N` | Stop after N frames | until `Ctrl-C` |
| `--duration S` | Stop after S seconds | until `Ctrl-C` |
| `--quiet` | Suppress per-frame lines; print the summary only | off |

The two groups:

| Stream | Group | Content |
| --- | --- | --- |
| Raw | `239.10.10.10:5002` | Sensor counts, always emitted |
| Calibrated | `239.10.10.11:5003` | W/m²/nm, emitted when the device carries coefficients |

{% hint style="warning" %}
On a machine with several network interfaces, set `--iface` to the IP of the NIC
the sensors are on. The OS default route is frequently not the right one, and
the symptom is simply **no frames** — no error.
{% endhint %}

## Telling sensors apart

Datagram **v2** (firmware 1.7.0+) carries the sender's MAC, sensor serial,
model, and per-frame integration time, so frames are self-describing and the
script demultiplexes on identity.

{% hint style="danger" %}
Older firmware emits **v1**, which carries none of that. Two v1 units on one
group are separable only by UDP source address — and a receiver that does not
filter reads a **~50/50 blend of both while looking perfectly healthy**. That is
not hypothetical; it happened on hardware on 2026-07-14.

`daq_stream.py` keys v1 frames on source IP and prints a warning, but the real
fix is updating the firmware.
{% endhint %}

A new sensor announces itself as it appears:

```
+ sensor 11-22-33-44-55 daq-e ptp from 192.168.1.50
+ sensor 192.168.1.51 from 192.168.1.51  [legacy v1: no identity in the frame, keyed on source IP. Update to fw 1.7.0+ for reliable multi-sensor separation.]
```

## Timestamps and PTP

`timestamp_us` is latched on the ESP32 as the sensor's last byte arrives, so it
excludes network and OS jitter.

When a frame reports PTP sync, clocks across sensors are disciplined to a common
grandmaster — typically ~50 µs on this hardware — and frames from different
units are directly comparable. That is what makes multi-sensor and
sensor-to-LATTICE alignment meaningful. Without PTP the stamps are only as good
as each device's own boot clock.

## Datagram fields

Each decoded frame carries:

| Field | Notes |
| --- | --- |
| `version` | 1 or 2 |
| `seq` | Per (device, stream) sequence number — used for drop detection |
| `timestamp_us` | Latched on the sensor |
| `absolute_time`, `ptp_synced`, `saturated`, `calibrated` | Header flags |
| `mac`, `sensor_serial`, `model`, `integration_time_ms` | v2 only; `None` on v1 |
| `spectrum` | Float array — counts, or W/m²/nm when calibrated |
| `device_key` | Sensor serial when present, else source IP |

Datagrams are validated on magic bytes, declared length, and a CRC-16/CCITT over
the header and payload; anything that fails is counted as malformed and skipped.
The full layout is in
[`PROTOCOL.md`](https://github.com/mapircamera/ESP32/blob/main/PROTOCOL.md).

## Summary output

On exit the script reports per-sensor frame and drop counts, derived from the
sequence numbers:

```
2 sensor(s), 1200 frame(s)
  11-22-33-44-55 daq-e ptp            frames=600     dropped=0 (0.00%)
  11-22-33-44-66 daq-e ptp            frames=600     dropped=3 (0.50%)
```

## Two meanings of "calibrated"

{% hint style="info" %}
`--calibrated` here is computed **by the device**. `--calibrate` in
`record_daq.py` applies the bundle **locally in Python**. Both produce the same
numbers — the device runs a pre-folded version of the same arithmetic — so use
whichever fits: the stream needs no calibration machinery on your side, the
local path works on any firmware.

{% content-ref url="offline-calibration.md" %}
[offline-calibration.md](offline-calibration.md)
{% endcontent-ref %}
{% endhint %}
