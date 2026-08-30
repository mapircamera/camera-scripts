---
description: >-
  The two ways attitude reaches a recording — the per-frame trailer and the
  standalone IMU stream — and what --imu-rate keeps.
---

# Attitude (IMU)

A DAQ-E-S carries an accelerometer, and attitude qualifies the irradiance beside
it: a tilted cosine corrector is not measuring the same hemisphere a level one
is. Attitude arrives **two ways**, and `daq_stream.py` records both.

## The per-frame trailer

On firmware **1.8.0+** — which is every DAQ-E-S — a spectral frame can carry a
22-byte attitude trailer appended *after* the CRC, announced by flags bit 4.

It lands in that reading's own `imu_*` columns, so the tilt sits beside the
irradiance it qualifies. Free — no extra rows — but limited to the ~2.5 Hz that
spectra arrive at.

{% hint style="danger" %}
The trailer sits **outside** the CRC deliberately, so a corrupt trailer can never
cost you a good spectrum. The corollary is that **a reader must tolerate the
extra bytes**. A reader that checks for an exact datagram length instead rejects
every frame such a unit sends, counts them as malformed, and shows the sensor as
absent while it streams perfectly.
{% endhint %}

`daq_stream.py` accepts them and reports the trailer's presence in the `imu` CSV
column, alongside `tilt_deg` / `roll_deg` / `pitch_deg`. It does not decode the
full attitude payload itself — that layout lives in
[`PROTOCOL.md`](https://github.com/mapircamera/ESP32/blob/main/PROTOCOL.md), and
Chloros decodes it.

## The standalone stream

On firmware **1.12+** the device publishes attitude on its own multicast group,
`239.10.10.12:5004`, at the accelerometer's rate (~50 Hz) — independent of
spectra. That is the only way to get attitude at better than the ~2.5 Hz the
per-frame trailer allows.

```bash
# the default: 5 Hz, small files
python daq_stream.py --imu --daq attitude.daq --csv attitude.csv
```

```bash
# everything the device sends
python daq_stream.py --imu --imu-rate 0 --daq attitude.daq
```

`--imu` records the samples as `event_type = 4` rows carrying **no spectrum**.

{% hint style="info" %}
Chloros's spectral readers never see them: `mip/daq_dls` predicates on
`event_type = 3 AND spectral_data IS NOT NULL`, and `mip/als.py` skips rows whose
blob will not decode. Nothing in Chloros reads them back *yet* — they are
recorded so the data exists on the same clock and in the same file as the
spectra, instead of stranded in a sidecar.
{% endhint %}

## `--imu-rate` decides how much you keep

It defaults to **5 Hz**. 50 Hz is ten times the rows in both outputs and more
than most work needs, so samples are thinned **by their own timestamps** — not by
a frame counter, which would drift whenever the device's actual rate did. Raise
it for vibration or fast-attitude work; `--imu-rate 0` keeps every sample.

Measured on a 50 Hz stream: 5 Hz keeps a tenth of the samples and about a tenth
of the CSV bytes.

## The attitude CSV

`--imu --csv` writes one row per kept sample, with a `#` provenance line naming
the group and units at the top:

| Columns | |
| --- | --- |
| `sensor_serial`, `model`, `seq` | Sender identity and per-stream sequence |
| `timestamp_us`, `ptp_synced` | Latched on the device; PTP flag |
| `sample_age_ms`, `fresh`, `healthy`, `angles_valid` | Sample freshness and validity |
| `cal_applied`, `tempco_applied`, `tare_set`, `tare_exceeded` | Which corrections the firmware applied |
| `tilt_deg`, `roll_deg`, `pitch_deg`, `tare_angle_deg` | The angles |
| `x_mg`, `y_mg`, `z_mg`, `mag_mg` | Corrected acceleration vector and magnitude |
| `raw_x_mg`, `raw_y_mg`, `raw_z_mg`, `temp_raw` | Uncorrected vector and die temperature |

{% hint style="warning" %}
An empty cell means **undefined**, never zero. An angle the firmware did not
compute is not "level" — writing `0.0` for it would be a measurement nobody made.
{% endhint %}

## What can be recorded, and to what

| Stream | `.daq` | `.csv` | How |
| --- | :---: | :---: | --- |
| Raw spectra | ✅ | ✅ | `record_daq.py <k> --csv` (any model) or `daq_stream.py --daq --csv` (DAQ-E) |
| Calibrated spectra | ✅ | ✅ | `daq_stream.py --calibrated --daq --csv`, or `record_daq.py e --calibrate bake`/`csv` |
| Attitude (IMU) | ✅ | ✅ | `daq_stream.py --imu --daq --csv`; the per-frame trailer also lands in a spectral `.daq`'s `imu_*` columns automatically |

{% content-ref url="daq-file-format.md" %}
[daq-file-format.md](daq-file-format.md)
{% endcontent-ref %}
