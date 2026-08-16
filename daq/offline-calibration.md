---
description: >-
  Getting calibrated spectral irradiance from a DAQ-E with no cloud, no account,
  and no Chloros — using the bundle stored on the device itself.
---

# Offline Calibration

A **DAQ-E** carries its own factory calibration bundle in flash. `daq_cal.py`
pulls it down over the JSON control channel and reproduces the same radiometric
transform Chloros applies at import, so you get spectral irradiance in
**W/m²/nm** with no AWS, no account, and no Chloros install.

{% hint style="info" %}
This is DAQ-E only. **DAQ-U** and **DAQ-M** have no onboard store (no
filesystem, no ethernet) — record raw and let Chloros calibrate at import, or
hand `daq_cal` a bundle from elsewhere:

```python
from daq_cal import DeviceCalibration
cal = DeviceCalibration.from_bundle(json.load(open("bundle.json")))
```
{% endhint %}

## Inspecting a device

```bash
python daq_cal.py 192.168.1.50
```

Records nothing; just prints the correction chain the unit will run:

```
sensor 11-22-33-44-55 kind=daq-e
  bundle    9f2a1c0b8e4d7a63... completed 2026-05-12T18:04:11Z
  points    135  (374-974 nm)
  dark      rate_plus_offset_over_t
  pi        applied at runtime
  cap       none (no profile -- uncorrected)  [profiles from: none]

  dark at   1 ms integration: 0.5183 raw units
  dark at  50 ms integration: 0.0142 raw units
  dark at 500 ms integration: 0.0091 raw units

  firmware 1.7.0
```

| Option | Meaning |
| --- | --- |
| `--control-port N` | Control TCP port (default `5001`) |
| `--token TOKEN` | Control-channel auth token, if the unit has one set |
| `--save-bundle PATH` | Also write the raw bundle JSON to disk |
| `--require-profiles` | Fail if the device has no cap/geometry profiles |

## Recording with calibration

```bash
# .daq stays RAW; calibrated irradiance goes to a sibling .csv
python record_daq.py e --host 192.168.1.50 --calibrate csv
```

| `--calibrate` | `.daq` contents | When to use |
| --- | --- | --- |
| `off` *(default)* | Raw counts | Chloros will calibrate at import |
| `csv` | Raw counts, **plus** a calibrated `.csv` | **Recommended.** Usable numbers now, and the `.daq` stays reprocessable |
| `bake` | Calibrated W/m²/nm, stamped with the bundle SHA | The recording must stand completely alone |

{% hint style="warning" %}
Prefer `csv` over `bake`. A raw `.daq` can be re-calibrated later if a
coefficient revision lands; a baked one is frozen at whatever the bundle said
the day you recorded.
{% endhint %}

The CSV header carries provenance — sensor, bundle SHA, completion date, cap,
dark model, and profile source — so the file stands alone. `--csv PATH` sets an
explicit path; the default is the `.daq` path with a `.csv` suffix.

## The transform

```
dark_eff(t) = rate + offset_per_ms / t        (per-unit decomposition)
            | dark_mean * ((1-f) + f/t)       (fleet-fraction fallback)
            | dark_mean                       (legacy fixed scalar)

out = (raw - dark_eff(t)) * gain
out = out * pi                                (unless baked into gain)
out = max(out, 0)
out = out * cap_correction                    (unless bare / as_recorded)
```

**Why π:** the factory gain is anchored to integrating-sphere *wall radiance*,
so the radiance → hemispherical-irradiance conversion (`E = π · L`) has to be
restored at apply time. Bundles regenerated with it already folded into the gain
declare that, and the step is skipped — so it is never applied twice.

**Why the zero clamp:** in bands with no light (NIR under fluorescents, deep
blue indoors) the value is pure dark-subtraction residue, and `gain × π` turns
each residual raw count into a few mW/m²/nm — enough to draw a visibly negative
tail.

**Dark model tiers:** the fixed read-offset component shrinks as `1/t` because
the sensor normalizes its reported spectrum by integration time. The fallback
tiers both reduce to `dark_mean` at t = 1 ms, so the difference only matters at
the 32–500 ms integrations auto-exposure actually uses in the field.

{% hint style="danger" %}
Always pass **this frame's own** `integration_time_ms`. Auto-exposure moves it
between 1 and 500 ms and the dark model is a function of it. Omitting it falls
back to the fixed-scalar dark, which over-subtracts at long integrations, biases
the result low, and clips near-floor bands to zero.
{% endhint %}

## Programmatic use

```python
from daq_cal import DeviceCalibration

cal = DeviceCalibration.from_device("192.168.1.50")
print(cal.describe())
watts = cal.apply(raw_counts, integration_time_ms=50)
```

`DeviceCalibration` exposes `wavelength_nm`, `n_points`, `bundle_sha`,
`completed_utc`, `sensor_id`, `cap_id`, `dark_model`, and `profiles_source` for
provenance, plus `effective_dark(t)` if you need the dark value alone.

{% hint style="info" %}
The transform is a deliberate mirror of Chloros's `daq/calibration_apply.py`.
Both operate on float32 through the same sequence of operations, so for the same
inputs they agree to the bit — `selftest.py` compares against captured reference
vectors.
{% endhint %}

## Caps

A cosine corrector or FOV cone has its own per-wavelength correction, and even a
**bare** DAQ-E has a geometry correction (the bare diffuser over-reads
directional light ~2×, while a sunshine cap is near-ideal). None of that is in
the calibration bundle — it lives in Chloros and is versioned separately.

So the cap is **not a device setting you can change from these scripts.** It is
resolved by Chloros for the unit and pushed into the device's profile store,
where `daq_cal.py` reads it.

| Situation | Result |
| --- | --- |
| Device carries the right profile | Correct output, matches Chloros exactly |
| Device carries **no** profile | **Bare-uncorrected** — ~2× off bare, ~11× off under a sunshine cap |
| Physical cap changed | Re-push from Chloros; the scripts cannot substitute another cap's curve |

### `--cap-id` is an override, not a cap selector

The device carries exactly **one** resolved profile, so the only accepted values
are:

* `as_recorded` — skip every per-wavelength profile (π only), or
* the cap the device already carries (a no-op override).

{% hint style="danger" %}
Naming a **different** cap is refused. There is no local copy of its correction
curve, and applying the stored one under another name would be silently wrong by
the sunshine-vs-bare factor. The override is validated at startup, not
mid-recording — failing there costs nothing; failing mid-recording loses the run.
{% endhint %}

### `--require-profiles`

Use it when a cap **is** fitted, to refuse outright rather than log a plausible
wrong number:

```bash
python record_daq.py e --host 192.168.1.50 --calibrate csv --require-profiles
```

Without it, a device carrying no profiles still records — with a warning on
stderr that the output is bare-uncorrected.

{% hint style="info" %}
Profiles are firmware **v1.6.0+**. Older firmware answers `get_profiles` with an
unknown-command error, which is not a failure — it just means bare-uncorrected
output, the same as every pre-v1.6.0 recording. Connect the unit to Chloros once
to push profiles.
{% endhint %}
