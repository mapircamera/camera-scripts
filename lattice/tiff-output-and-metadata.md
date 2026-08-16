---
description: >-
  The filename pattern, EXIF tags, and pixel format Chloros reads when it
  imports a raw LATTICE TIFF written by these scripts.
---

# TIFF Output & Metadata

Everything on this page is written by `write_lattice_raw_tiff()` and
`lattice_capture_filename()` in `mapir_metadata.py`. That module is the
**Chloros ingest contract** — if you write your own capture loop, match it
exactly.

## Filename

```
capture_<serial>_<seq>_<YYYYMMDD>_<HHMMSS>_<subsec>.tiff
```

For example: `capture_213602328_0004_20260816_142233_507118.tiff`

* `serial` — digits of the camera serial.
* `seq` — 4-digit trigger sequence. **Every camera in one synchronized capture
  must share the same value**; Chloros groups a multi-camera shot on the part of
  the filename after the serial.
* `YYYYMMDD_HHMMSS_subsec` — capture time, with microseconds.

## Pixels

| Model | Content |
| --- | --- |
| **M3C (color)** | The **raw Bayer mosaic**, single channel, RGGB |
| **M3M (mono)** | The single-channel mono frame |

Both are 2-D `(H, W)` `uint16` arrays holding the sensor's native bit depth
(12-bit, 0–4095) left in a uint16 container.

{% hint style="danger" %}
Do **not** debayer and do not stack channels. Chloros debayers on import, and
the writer rejects anything that is not 2-D.
{% endhint %}

## EXIF tags

| Tag | Code | IFD | Value |
| --- | --- | --- | --- |
| `Make` | 0x010F | IFD0 | `MAPIR` |
| `Model` | 0x0110 | IFD0 | e.g. `LATT-M3C-L41-FRGN` — **the LATTICE detection key** |
| `CameraSerialNumber` | 0xC62F | IFD0 | Camera serial — **the calibration fetch key** |
| `ExposureTime` | 0x829A | EXIF sub-IFD | Seconds (rational) |
| `ISOSpeedRatings` | 0x8827 | EXIF sub-IFD | ISO equivalent; Chloros derives gain as `20·log10(ISO/100)` |
| `DateTimeOriginal` | 0x9003 | EXIF sub-IFD | Naive wall-clock stamp |
| `OffsetTimeOriginal` | 0x9011 | EXIF sub-IFD | Timezone of the stamp above, e.g. `+00:00` |
| `SubSecTimeOriginal` | 0x9291 | EXIF sub-IFD | Microseconds |

The model string **must** start with `LATT-`; the writer raises if it does not,
because that prefix is how Chloros detects LATTICE images at all.

## Timezone contract

`DateTimeOriginal` is a naive wall-clock string, so on its own it is ambiguous.
`OffsetTimeOriginal` **declares** its zone, which makes the image side
self-describing.

The capture script stamps **UTC** everywhere, so the declaration is `+00:00`.
This is the same contract the MAPIR CM5 hub stamps, and it is what lets
image ↔ `.daq` reflectance matching work on any processing host with no manual
*Light sensor timezone offset* setting.

{% hint style="warning" %}
If you adapt the scripts to stamp **local** time, both declarations have to move
together — the EXIF `OffsetTimeOriginal` here and the `.daq`'s
`als_meta.utc_offset_minutes`. A naive `when` passed to
`write_lattice_raw_tiff()` is taken as host-local and declared as such.
{% endhint %}

{% content-ref url="../daq/daq-file-format.md" %}
[daq-file-format.md](../daq/daq-file-format.md)
{% endcontent-ref %}

## Compression

Frames are written **uncompressed** — about 6.3 MB for a full-res M3C/M3M
frame.

That is not an oversight. Pillow's in-TIFF DEFLATE/LZW paths go through
libtiff, which cannot co-write the EXIF sub-IFD this contract needs (libtiff
drops tag 34665, the pointer to the sub-IFD carrying exposure, ISO, and the
timestamps). Keeping the pixels uncompressed guarantees the metadata survives on
every platform.

Compress at rest (zip, or filesystem compression) if storage is tight — Chloros
reads compressed or not.

## Verifying your output

`selftest.py` round-trips everything these scripts write through verbatim copies
of the actual Chloros reader functions, and also checks the DAQ wire codec and
camera configuration. Run it any time:

```bash
python selftest.py
```
