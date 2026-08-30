---
description: >-
  Every recording declares which cap was fitted and who decided it — the one
  piece of provenance that keeps a 20–30× downwelling error from looking
  verified.
---

# Caps & Cap Provenance

A cosine corrector or FOV cone has its own per-wavelength correction, and even a
**bare** DAQ-E has a geometry correction (the bare diffuser over-reads
directional light ~2×, while a sunshine cap is near-ideal). None of that is in
the calibration bundle — it lives in Chloros and is versioned separately.

## Every recording declares a cap, and says who decided it

On DAQ-U / DAQ-M / DAQ-E the sunshine corrector is **removable**, and nothing on
the sensor can sense whether it is fitted — so somebody has to say.

`record_daq.py` assumes `sunshine_cosine`, because that is how the large
majority of units fly and it is the same assumption Chloros's own recorder
makes. A **DAQ-E-S** gets `as_recorded` instead, and that one is not an
assumption.

| Model | Default `cap_id` | Why |
| --- | --- | --- |
| DAQ-U | `sunshine_cosine` | Removable cap, fleet default |
| DAQ-M | `sunshine_cosine` | Removable cap, fleet default |
| DAQ-E | `sunshine_cosine` | Removable cap, fleet default |
| DAQ-E-S | `as_recorded` | Permanent diffuser — settled by the hardware |

The file records **which of those it was**, in `als_meta.cap_id_source`:

| Value | Meaning |
| --- | --- |
| `auto_default` | **Assumed** — nobody said, so the fleet default was used |
| `operator` | You stated it with `--cap-id` |
| `device` | Read back from the unit's own profile store |
| `model` | Settled by the hardware (DAQ-E-S) |

{% hint style="info" %}
That distinction is what makes an assumed cap **undoable**. Chloros warns on
`auto_default` — naming the file — and an operator can override the cap per
project and get the corrected number back, because a raw recording has not had
the cap multiplied in yet. A file that could not say whether anyone checked
would leave a 20–30× error looking exactly like a verified one.
{% endhint %}

The cap is recorded as *provenance* — `cap_applied = 0` — and Chloros applies it
at import.

## Declaring bare on a capped sensor

Pass `--cap-id none` **only** for a sensor you have physically stripped.
Declaring bare on a capped unit is not "uncorrected", it is wrong by the whole
correction, and nothing downstream can detect it:

| Model | `sunshine_cosine` | Declaring `none` instead |
| --- | --- | --- |
| DAQ-U | ×30.6 | **~30× low** |
| DAQ-M | ×23.1 | **~23× low** |
| DAQ-E | ×11.0 | **~22.6× low** — on a DAQ-E `none` is an *active* ×0.49 bare-geometry profile, not a no-op |

`record_daq.py` prints the cap it declared on every run, and warns when that is
`none`.

## The DAQ-E-S exception

None of the above applies to a **DAQ-E-S** — the one model whose diffuser really
is permanent. It was on the unit when its factory bundle was measured, so the
correction is already inside the gain. Chloros resolves that unit to
`as_recorded` (no per-wavelength profile of any kind) and **overrides** any cap
you ask for, because there is no cap choice to respect on optics that do not
come off.

{% hint style="danger" %}
Putting `sunshine_cosine` on top of a DAQ-E-S double-counts the diffuser:
measured at ~11×, which integrates to 6× the solar constant above the
atmosphere — physically impossible, and the tell that it has happened.
{% endhint %}

## `--cap-id` means two different things

* **With `--calibrate` (DAQ-E)** it is an override of the profile the device
  carries, and the device carries exactly one — so the only accepted values are
  `as_recorded` (skip every per-wavelength profile) or that same cap. Naming a
  different one is **refused**: there is no local copy of its curve, and applying
  the stored one under another name is silently wrong by the sunshine-vs-bare
  factor. Validated at startup, not mid-recording.
* **Without `--calibrate`** (every raw recording, and the only mode DAQ-U /
  DAQ-M have) nothing is applied locally, so `--cap-id` simply *declares* what is
  fitted, for Chloros to apply at import. Any id Chloros knows for that device
  kind is accepted — `sunshine_cosine` (the default), `none`,
  `fov_15`/`45`/`90`, and `fov_30`/`60` on DAQ-U.

## Setting the cap on a DAQ-E

A DAQ-E stores one resolved cap profile and folds it into the calibrated stream
it publishes, so that store is what decides whether an offline consumer of that
stream gets the truth. You can write it with `daq_cal.py`:

```bash
# see what the unit is currently applying
python daq_cal.py 192.168.1.50
```

```bash
# save the document it holds (the curve, verbatim)
python daq_cal.py 192.168.1.50 --save-profiles aboard.json
```

```bash
# apply NO per-wavelength profile at all -- needs no curve
python daq_cal.py 192.168.1.50 --set-cap as_recorded
```

```bash
# fit a different cap: supply that cap's curve, which Chloros ships
python daq_cal.py 192.168.1.50 --set-cap fov_45 --cap-profile /path/to/chloros/daq/cap_profiles/e/fov_45.json
```

The curve has to come from somewhere — the device applies what it is given and
carries no library to look one up in, so **every id except `as_recorded` needs
its `--cap-profile` JSON**. Chloros ships them at
`daq/cap_profiles/<kind>/<cap_id>.json`; copy the one you need. Asking for a real
cap without its curve is refused rather than written as bare.

Requires firmware **1.6.0+** (`set_profiles`). The document written is
byte-identical to the one Chloros builds for the same cap, so setting a cap here
does not make Chloros rewrite the unit's flash on its next connect.

{% hint style="warning" %}
**Chloros still has the last word.** It re-pushes its own resolved cap whenever
it connects and finds a different document. If the cap must survive, set it in
Chloros as well — this is for units that never meet a Chloros install, or for
checking what one is carrying.
{% endhint %}

| Situation | Result |
| --- | --- |
| Device carries the right profile | Correct output, matches Chloros exactly |
| Device carries **no** profile | **Bare-uncorrected** — ~2× off bare, ~11× off under a sunshine cap |
| Physical cap changed | Re-push from Chloros, or `--set-cap` with that cap's curve |

## What each mode stamps

The two flags Chloros reads are `calibration_applied` (are these W/m²/nm or
counts?) and `cap_applied` (is a cap correction already multiplied in?). Between
them they make every combination unambiguous, so nothing is ever calibrated or
capped twice:

| Run | `calibration_applied` | `cap_id` | `cap_applied` | Chloros applies |
| --- | :---: | --- | :---: | --- |
| `record_daq.py u/m/e` (default) | `0` | `sunshine_cosine` | `0` | Bundle **+** that cap |
| …on a DAQ-E-S | `0` | `as_recorded` | `0` | Bundle only, no profile |
| …`--cap-id none` (stripped) | `0` | `none` | `0` | Bundle **+** bare geometry (DAQ-E) or nothing (U/M) |
| `--calibrate csv` | `0` | The device's cap | `0` | Bundle + that cap — and the sibling `.csv` already has both |
| `--calibrate bake` | `1` | The device's cap | `1` | **Nothing** — imported as-is |
| `--calibrate bake --cap-id as_recorded` | `1` | `as_recorded` | `0` | Nothing; a later cap override can still be applied |

The last row is why `cap_applied` follows the cap actually handed to the
calibrator rather than merely whether the device had profiles: `as_recorded`
skips every profile, so stamping it as applied would claim a correction that is
not in the data — and Chloros would then *refuse* a later operator override (it
cannot undo a curve that was never applied) instead of correcting the file.

## `--require-profiles`

Use it to refuse outright rather than log a plausible wrong number when a cap is
fitted but no profile is aboard:

```bash
python record_daq.py e --host 192.168.1.50 --calibrate csv --require-profiles
```

{% content-ref url="offline-calibration.md" %}
[offline-calibration.md](offline-calibration.md)
{% endcontent-ref %}
