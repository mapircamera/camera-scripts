---
description: >-
  Python scripts to control MAPIR cameras and light sensors and to process the
  images they produce.
---

# MAPIR Scripts

Reference Python scripts for MAPIR hardware. They are deliberately small,
dependency-light, and heavily commented — meant to be read, copied, and adapted
for DIY drone and research setups.

## LATTICE & DAQ — capture your own raw data

The [**LATTICE**](lattice/overview.md) and [**DAQ**](daq/overview.md) sections
document the scripts in
[`mapircamera/chloros_scripts`](https://github.com/mapircamera/chloros_scripts):
capture raw frames from LATTICE cameras and raw spectra from DAQ light sensors
**with your own computer**, then process them in
[Chloros](https://www.mapir.camera/collections/software/products/chloros)
afterward.

{% hint style="info" %}
**The idea:** you capture **raw** data in the field; **Chloros calibrates and
processes it later**. The scripts apply *no* processing and *no* calibration.
They stamp each file with the device's **serial number and model**, and on
import Chloros fetches that exact device's factory calibration from the cloud
and applies it. Capture is yours; the science is handled at import.

And Chloros **hands the result back**: every `.daq` you import is written out
again, calibrated, as a `.daq` *and* a `.csv` of spectral irradiance — for
**DAQ-U, DAQ-M and DAQ-E alike**, with no camera involved. See
[Light Sensor Only](daq/light-sensor-only.md).
{% endhint %}

| Script | Purpose |
| --- | --- |
| `capture_lattice.py` | Control + raw capture from **LATTICE cameras** (M3C/M3M), with hardware-cable multi-camera sync |
| `record_daq.py` | Record raw spectra from a **DAQ-U / DAQ-M / DAQ-E** to a Chloros-compatible `.daq` |
| `daq_stream.py` | Listen to any number of **DAQ-E / DAQ-E-S** sensors over multicast — raw, calibrated, or attitude — and record what arrives |
| `daq_cal.py` | Read or write a **DAQ-E**'s onboard factory calibration and cap profile — no cloud, no Chloros |
| `mapir_metadata.py` | The Chloros ingest contract: writes raw LATTICE TIFFs + the `.daq` SQLite format |
| `selftest.py` | Self-contained checks that the output matches what Chloros reads on import |

## Survey3 — post-processing

The [**Survey3**](survey3/convert-raw-jpg-to-tiff.md) section covers the batch
post-processing scripts in this repository, plus triggering a Survey3 from a
Raspberry Pi:

* **`Convert_Survey3_RAW_to_Tiff/`** — converts the Survey3 RAW+JPG pair into
  TIFF images, with optional white balance, carrying EXIF across.
* **`Vignette_Correction_Survey3/`** — applies the flat-field (vignette)
  correction to Survey3 images (S3N and S3W flat fields included).

Each folder ships a `.bat` launcher alongside the Python script.

## License & support

MIT licensed (see [LICENSE](LICENSE)); provided as-is, without warranty. For
help with MAPIR hardware or Chloros, contact
[MAPIR support](https://www.mapir.camera/community/contact).
