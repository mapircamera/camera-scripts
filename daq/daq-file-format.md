---
description: >-
  The .daq SQLite schema Chloros imports — als_meta, als_log, and the timezone
  contract that makes image matching work.
---

# The .daq File Format

A `.daq` is a **SQLite database** with two tables, written by `DaqWriter` in
`mapir_metadata.py`. If you write your own recorder, match this exactly — it is
the Chloros ingest contract.

## `als_meta` — one row, device identity

| Column | Value written |
| --- | --- |
| `version` | `1.27` — the schema number, matching the MAPIR hub's |
| `product_model` | `daq-u` / `daq-m` / `daq-e` — the device kind |
| `product_serial` | **The calibration fetch key.** Read from the sensor |
| `device_name` | Free-text label (optional) |
| `calibration_applied` | `0` for raw (default), `1` when values are already calibrated |
| `calibration_bundle_sha` | SHA-256 of the bundle that produced calibrated values |
| `calibration_completed_utc` | The bundle's own completion date, for audit |
| `cap_id` | Cap-correction profile id, or `none` for a bare sensor |
| `cap_applied` | `1` when a cap profile was folded into the stored spectra |
| `cap_id_source` | **Who decided `cap_id`** — `operator` / `device` / `model` / `auto_default` |
| `utc_offset_minutes` | Timezone declaration — `0` for these scripts (schema v1.23+) |

{% hint style="warning" %}
`cap_id_source` is what makes an assumed cap undoable: Chloros warns on
`auto_default` and lets an operator override the cap per project, which is
impossible if the file cannot say whether anyone ever looked at the sensor. Never
stamp `operator` for a guess.

{% content-ref url="caps.md" %}
[caps.md](caps.md)
{% endcontent-ref %}
{% endhint %}

{% hint style="danger" %}
`calibration_applied = 1` **requires** `calibration_bundle_sha`. `DaqWriter`
raises otherwise: an applied calibration with no provenance cannot be audited or
reprocessed.
{% endhint %}

## `als_log` — one row per reading

| Column | Notes |
| --- | --- |
| `id` | Autoincrement primary key |
| `created_on` | Row insert time, SQLite default |
| `precise_timestamp` | **Absolute** wall-clock nanoseconds since the Unix epoch |
| `event_type` | `3` for a spectrum reading |
| `spectral_data` | The spectrum BLOB (see below) |
| `is_saturated` | `0` / `1` |
| `integration_time` | Integration time for **this** frame, in ms |
| `device_ts_ns`, `device_ts_ptp` | The device's own latched timestamp, and whether it was PTP-disciplined |

The full schema matches the MAPIR recorder. A DIY raw recorder fills only
`event_type` / `precise_timestamp` / `spectral_data` / `is_saturated` /
`integration_time`; the photometric columns stay `NULL` — Chloros recomputes them
from the calibrated spectrum at import.

{% hint style="info" %}
A `.daq` is a log, not a spectrum table: Chloros's readers select spectra
explicitly — `mip/daq_dls` predicates on `event_type = 3 AND spectral_data IS NOT
NULL`, and `mip/als.py` skips rows whose blob will not decode — so a row that is
not a spectrum reading is inert to all of them.
{% endhint %}

### The spectrum BLOB

Raw bytes of `np.save` for a **float32** array. Import does
`np.load(BytesIO(blob))`.

```python
buf = io.BytesIO()
np.save(buf, np.asarray(spectrum, dtype=np.float32))
blob = buf.getvalue()
```

{% hint style="warning" %}
`precise_timestamp` must be an **absolute** epoch — `time.time_ns()`, never a
monotonic clock. Chloros matches the DAQ to imagery by absolute time.
{% endhint %}

## Writing one

```python
from mapir_metadata import DaqWriter

w = DaqWriter("flight.daq", product_model="daq-u",
              product_serial=sensor_serial)
for spectrum, saturated, integration_ms in readings:
    w.write(spectrum, saturated, integration_ms, timestamp_ns=time.time_ns())
w.close()
```

`DaqWriter` also works as a context manager. Each instance starts a **fresh**
recording — an existing file at that path is replaced, since the tables cannot
be created over an existing database. Rows are committed every 10 readings and
on `close()`.

## Timezone contract

`utc_offset_minutes` (schema v1.23) declares the UTC offset, in signed minutes,
of the **naive wall-clock stamps** your capture system writes — filenames, EXIF
`DateTime`. It is the one field Chloros checks to line a `.daq` up with imagery
on any processing host, with **no** manual *Light sensor timezone offset*
setting.

These scripts stamp UTC everywhere, so they write `0` — the same convention the
MAPIR CM5 hub stamps.

{% hint style="warning" %}
If you adapt the scripts to stamp **local** time, both declarations have to move
together: this field (pass `utc_offset_minutes()` from `mapir_metadata`) and the
LATTICE side's EXIF `OffsetTimeOriginal`. The spectrum timestamps themselves are
absolute epochs and are not affected.

{% content-ref url="../lattice/tiff-output-and-metadata.md" %}
[tiff-output-and-metadata.md](../lattice/tiff-output-and-metadata.md)
{% endcontent-ref %}
{% endhint %}

## How Chloros uses the file

* **Serial number is the key.** Chloros looks up that exact device's factory
  calibration in the cloud and applies it at import.
* **Raw means raw.** Spectra are the sensor's raw firmware output;
  `calibration_applied = 0` tells Chloros to calibrate on import. The one
  exception is `--calibrate bake`, which sets `1` and carries the bundle SHA —
  Chloros then imports those spectra as-is instead of calibrating them a second
  time.
* **Reflectance needs downwelling + a synced clock.** Record a DAQ during the
  flight and keep the host clock accurate. With no DAQ you still get radiance,
  not reflectance.
* **You get the calibration back, not just its effect.** Importing a `.daq`
  writes `<project>/Light Sensor/<name>_calibrated.daq` and `.csv` — the
  calibrated spectra as a file, rather than only as an intermediate on the way to
  reflectance. No camera required.

{% content-ref url="light-sensor-only.md" %}
[light-sensor-only.md](light-sensor-only.md)
{% endcontent-ref %}

## Verifying your output

`selftest.py` round-trips everything these scripts write through verbatim copies
of the actual Chloros reader functions:

```bash
python selftest.py
```
