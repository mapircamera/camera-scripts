---
description: >-
  Python scripts to process images from MAPIR Survey3 cameras.
---

# MAPIR Survey3 Scripts

Reference Python scripts for MAPIR **Survey3** cameras. They are deliberately
small, dependency-light, and heavily commented — meant to be read, copied, and
adapted for DIY drone and research setups.

{% hint style="info" %}
**Looking for LATTICE or DAQ?** The capture scripts for LATTICE cameras and DAQ
light sensors live in
[`mapircamera/chloros_scripts`](https://github.com/mapircamera/chloros_scripts).
This repository covers Survey3 only.
{% endhint %}

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

MIT licensed (see [LICENSE](LICENSE)); provided as-is, without warranty.
Versions of this repository before November 2026 were distributed under
GPL-3.0. The bundled `exiftool.exe` is Phil Harvey's
[ExifTool](https://exiftool.org/), distributed under its own license (the same
terms as Perl). For help with MAPIR hardware or Chloros, contact
[MAPIR support](https://www.mapir.camera/community/contact).
