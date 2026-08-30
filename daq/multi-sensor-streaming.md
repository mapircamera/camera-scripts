---
description: >-
  Reading any number of DAQ-E / DAQ-E-S sensors at once over UDP multicast with
  daq_stream.py, recording what arrives, and why sensor identity in the datagram
  matters.
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

```bash
# record the stream to a Chloros-compatible .daq as well
python daq_stream.py --calibrated --daq flight.daq --csv flight.csv
```

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--calibrated` | Listen to the calibrated stream instead of raw counts | raw |
| `--imu` | Listen to the standalone [attitude stream](attitude-imu.md) instead of spectra | off |
| `--imu-rate HZ` | With `--imu`: attitude samples per second to keep; `0` keeps all | `5` |
| `--group` / `--port` | Override the multicast group / port | per stream |
| `--iface IP` | Local interface IP to join on | OS default |
| `--serial S` | Only show this sensor serial; repeatable | all |
| `--csv PATH` | Append every frame to a CSV | none |
| `--daq PATH` | Also record to a Chloros-compatible `.daq` | none |
| `--serial-model M` | With `--daq`: override the `product_model` stamped | what the frames report |
| `--cap-id ID` | With `--daq` on a **calibrated** stream: which cap the device folded in | `as_recorded` |
| `--count N` | Stop after N frames | until `Ctrl-C` |
| `--duration S` | Stop after S seconds | until `Ctrl-C` |
| `--quiet` | Suppress per-frame lines; print the summary only | off |

The three groups:

| Stream | Group | Content |
| --- | --- | --- |
| Raw | `239.10.10.10:5002` | Sensor counts, always emitted |
| Calibrated | `239.10.10.11:5003` | W/m²/nm, emitted when the device carries coefficients — a DAQ-E-S always does |
| Attitude | `239.10.10.12:5004` | Attitude at the accelerometer's rate, firmware 1.12+ |

{% hint style="warning" %}
On a machine with several network interfaces, set `--iface` to the IP of the NIC
the sensors are on. The OS default route is frequently not the right one, and
the symptom is simply **no frames** — no error.
{% endhint %}

## Recording what arrives

`--daq` writes the stream to the same `.daq` format `record_daq.py` produces, so
a multicast recording imports into Chloros exactly like a TCP one.

The file declares **what the stream actually carried**: a calibrated stream is
stamped `calibration_applied = 1` so Chloros imports it as-is instead of applying
its bundle a second time. That comes from the frame's own flag bit, not from
which group the script joined.

With `--calibrated --daq`, tell the file which cap the device folded in with
`--cap-id` — read it off the unit with `daq_cal.py`. The default is
`as_recorded`, because the device applied whatever it holds and the script is not
guessing which:

{% content-ref url="caps.md" %}
[caps.md](caps.md)
{% endcontent-ref %}

{% hint style="warning" %}
The calibrated stream is folded on the device from whatever profile document
Chloros last pushed. A unit carrying a stale or wrong-cap document emits a
plausible number that is out by the whole geometry correction — up to ~11× for a
sunshine cap. `daq_cal.py <host>` prints exactly what is aboard.
{% endhint %}

## Every frame says which stream it came from

Raw counts and calibrated W/m²/nm differ by roughly four orders of magnitude, and
nothing about the numbers themselves announces which you are holding. So the CSV
carries a `calibrated` column, a `units` column and an `imu` column per row —
taken from the **frame's own flag bit**, not from the group the script joined —
plus a `#` provenance line naming the group and units at the top.

{% hint style="danger" %}
If a frame's flag ever contradicts its group — a firmware bug, since the two
groups are meant to be exclusive — the script warns on stderr and records **what
actually arrived**, not what the group implied.
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
| `has_imu`, `imu` | Whether a 22-byte attitude trailer was appended, and its decoded angles |
| `device_key` | Sensor serial when present, else source IP |

Datagrams are validated on magic bytes, declared length, and a CRC-16/CCITT over
the header and payload; anything that fails is counted as malformed and skipped.
The full layout is in
[`PROTOCOL.md`](https://github.com/mapircamera/ESP32/blob/main/PROTOCOL.md).

{% hint style="danger" %}
A firmware 1.8.0+ frame may carry a 22-byte attitude trailer **after** the CRC.
A reader that checks for an exact datagram length rejects every frame such a unit
sends, counts them as malformed, and shows the sensor as absent while it streams
perfectly. `daq_stream.py` tolerates the extra bytes:

{% content-ref url="attitude-imu.md" %}
[attitude-imu.md](attitude-imu.md)
{% endcontent-ref %}
{% endhint %}

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
