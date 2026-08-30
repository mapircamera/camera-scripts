---
description: >-
  Recording a light sensor with no camera at all, and getting the calibrated
  .daq + .csv back out of Chloros.
---

# Light Sensor Only

Plenty of setups are **just a light sensor**: no camera, no imagery, no
reflectance — you want spectral irradiance, PPFD or lux over time, calibrated.
That is a first-class workflow, and a project with no images in it is a valid
project.

Capture raw in the field, then let Chloros calibrate:

```bash
python record_daq.py u --port COM7 --duration 3600
```

## Importing

Take the `.daq` files to any one of:

* **Desktop app** — make a project, drag the `.daq` files in, press **Process**.
  Nothing camera-specific to configure.
* **CLI** — `chloros-cli process ./my_daq_files -o ./my_project`
* **SDK** — see below.

{% hint style="warning" %}
The **CLI and SDK need a paid Chloros+ plan** (enforced server-side). The
desktop app route does not — it works on the free tier.
{% endhint %}

## What you get back

Chloros fetches each sensor's factory calibration **by serial** (local cache
first, then the MAPIR cloud), applies it, and writes two products per recording
into a `Light Sensor` folder inside the project:

```
<project>/
└── Light Sensor/
    ├── u_20260826_143012_calibrated.daq   # reprocessable archive
    └── u_20260826_143012_calibrated.csv   # W/m²/nm + photometrics
```

The `.csv` is one row per reading:

| Columns | |
| --- | --- |
| `timestamp_utc`, `timestamp_ns` | Absolute UTC epoch of the reading |
| `integration_time_ms`, `calibrated` | Acquisition + per-frame calibration flag |
| `total_power_W_m2` | Integrated irradiance |
| `photopic_lux`, `scotopic_lux` | Photometric illuminance |
| `ppfd_umol_m2_s`, `ppfd_blue`, `ppfd_green`, `ppfd_red` | PAR photon flux, total and split |
| `peak_wavelength_nm` | Spectral peak |
| `340.0` … `1010.0` | The full spectrum, W/m²/nm, 135 points at 5 nm |

The header rows above the columns carry provenance — sensor, kind, bundle SHA,
completion date, cap, dropped-saturated count, source filename, and `clock=`,
which names the time axis the frames are on.

The `.daq` beside it is the same SQLite format these scripts write, now carrying
calibrated spectra and declaring the bundle that produced them — so re-importing
it does **not** calibrate it a second time.

## From the SDK

```python
import chloros_sdk

with chloros_sdk.ChlorosLocal() as cl:
    cl.create_project("DAQ-U_2026-08-26")
    cl.import_images("./my_daq_files")     # .daq files; no imagery needed
    result = cl.export_light_sensor()

for rec in result["exported"]:
    print(rec["csv"])
for rec in result["skipped"]:
    print("skipped", rec["source"], "--", rec["reason"])
```

## Raw counts are named `_raw`, not `_calibrated`

One sensor family has no per-serial calibration bundle and never will: the
**DAQ-A**, which predates the DAQ-U/M/E bundle system. It is not obsolete — it is
flown with a reflectance target and a Survey3 today, and Chloros calibrates with
it, because only *relative* response matters for solving the target quadratics.

Those recordings export, but as what they are:

* the products are named **`<stem>_raw.daq`** and **`<stem>_raw.csv`**, not
  `_calibrated` — a different filename rather than a flag inside the file,
  because the claim has to survive being emailed on as a bare name;
* the CSV header says `raw spectral sensor counts (NOT irradiance)`, and warns
  the values are comparable **within** the file and not across sensors;
* the run summary says `exported (N calibrated, M as RAW COUNTS)`, so
  "exported" in a log cannot be read as irradiance;
* the power-dependent photometric columns stay `NULL`, and the `.daq` keeps
  `calibration_applied = 0`, so re-importing one cannot double-apply anything.

{% hint style="info" %}
This applies **only** to a family that is known to carry no bundle. An unknown or
blank model is *not* assumed bundle-less — erring that way would turn a real
fetch failure into a silent raw export.
{% endhint %}

## A recording that can't be calibrated is skipped, not faked

If you are offline, or that serial has no calibration on file, Chloros reports
the recording under `skipped` **with the reason the file itself gives** and
writes nothing for it — rather than emitting a `*_calibrated.csv` that holds raw
counts. Reconnect, re-run, and it completes.

The reason is read off the recording rather than assumed: a run summary lists the
actual causes, deduplicated, so N files skipped for one cause read as one cause
instead of a blanket guess about uncached bundles.

## Legacy recordings

Older `.daq` schemas (v1.01 / v1.02) carry no `precise_timestamp` column — only
the row write time. The downwelling matcher still refuses them, correctly, since
matching imagery needs a per-exposure epoch. **Export does not**: those files are
read on the `created_on` axis and the product states which clock it is on, via
`clock=daq_created_on` in the CSV header.

## Your raw file is never modified

The products are written alongside it. Keep the raw one — it is the master, and
stays re-calibratable against a future coefficient revision.

{% content-ref url="daq-file-format.md" %}
[daq-file-format.md](daq-file-format.md)
{% endcontent-ref %}
