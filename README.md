# False Color LUT Generator

A Python script that generates `.cube` LUT files for false color exposure monitoring across multiple camera log profiles. Pre-made LUTs for all supported profiles are included in the `luts/` folder — no script required for basic use.

## Pre-made LUTs

Ready-to-use files are in the `luts/` folder:

| File | Profile |
|------|---------|
| `FalseColor_vlog.cube` | Panasonic V-Log L |
| `FalseColor_slog3.cube` | Sony S-Log3 |
| `FalseColor_logc3.cube` | ARRI LogC3 |
| `FalseColor_clog2.cube` | Canon C-Log2 |
| `FalseColor_flog2.cube` | Fuji F-Log2 |
| `FalseColor_nlog.cube` | Nikon N-Log |
| `FalseColor_bmpfilm5.cube` | Blackmagic Film Gen 5 |

Drop the file matching your camera's log profile into DaVinci Resolve, Premiere Pro, or your camera's LUT slot.

## Supported Profiles

| Key | Profile | Cameras |
|-----|---------|---------|
| `vlog` | Panasonic V-Log L | GH5, GH6, S5, S5II, S1, BGH1, AU-EVA1 |
| `slog3` | Sony S-Log3 | A7S III, A7 IV, FX3, FX6, FX9, ZV-E1, Venice |
| `logc3` | ARRI LogC3 (EI800) | Alexa Mini, Alexa Mini LF, Alexa 35, Amira |
| `clog2` | Canon C-Log2 | C70, C300 III, C500 II, EOS R5 C |
| `flog2` | Fuji F-Log2 | X-H2S, X-H2, GFX100S II |
| `nlog` | Nikon N-Log | Z6, Z7, Z6II, Z7II, Z8, Z9 |
| `bmpfilm5` | Blackmagic Film Gen 5 | Pocket 6K G2, 6K Pro, URSA Mini Pro 12K |

## Default Zone Map

| Zone | Stops | Color |
|------|-------|-------|
| White Clip | clipping | Red `#ef4444` |
| +2 Stops | +2 | Yellow `#eab308` |
| +1 Stop | +1 | Magenta `#d946ef` |
| Mid Gray | 0 | Green `#22c55e` |
| -1 Stop | -1 | Cyan `#06b6d4` |
| -2 Stops | -2 | Blue `#3b82f6` |
| Black Clip | crushed blacks | Indigo `#6366f1` |
| Between zones | — | Grayscale passthrough |

Zones are defined relative to each profile's middle gray — the same stop values map to different log-encoded values depending on the profile.

## Requirements

- Python 3.11+
- NumPy

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install numpy
```

## Usage

**Generate a single profile** (set `LOG_PROFILE` in the CONFIG section first):

```bash
python false_color_lut_generator.py
```

**Regenerate all 7 profiles at once:**

```bash
python generate_all.py
```

Output files are written to the `luts/` folder.

## Configuration

All settings are in the `CONFIG` section at the top of `false_color_lut_generator.py`.

### Profile

```python
LOG_PROFILE = "vlog"   # vlog | slog3 | logc3 | clog2 | flog2 | nlog | bmpfilm5
```

### LUT Size

```python
LUT_SIZE = 33   # 33 = standard, 65 = higher quality (larger file)
```

### Zone Width and Blend

```python
ZONE_HALF_WIDTH_STOPS = 0.30   # width of each color band (in stops)
BLEND_WIDTH_STOPS     = 0.12   # softness at zone edges (in stops)
```

### Custom Zones

Each zone is defined as `(stop_value, "hex_color")`. Use `"white_clip"` and `"black_clip"` for the clipping zones.

```python
ZONES = [
    ("white_clip",  "#ef4444"),   # Clipping highlights
    ( 2,            "#eab308"),   # +2 Stops
    ( 1,            "#d946ef"),   # +1 Stop
    ( 0,            "#22c55e"),   # Mid Gray
    (-1,            "#06b6d4"),   # -1 Stop
    (-2,            "#3b82f6"),   # -2 Stops
    ("black_clip",  "#6366f1"),   # Crushed blacks
]
```

Stop values are relative to each profile's middle gray. Anything outside all zones renders as grayscale.

## Using the Output LUT

- **DaVinci Resolve** — Color page > LUTs > Apply as a node
- **Premiere Pro** — Lumetri Color > Creative > Look
- **Camera** — Load into a LUT slot via the camera menu (check your camera's manual for supported format and size)

## How It Works

1. Stop values in `ZONES` are converted to log-encoded values using each profile's transfer function (V-Log, S-Log3, LogC3, etc.).
2. For each point in the 3D color cube, perceptual luminance is calculated using Rec.709 weights (`0.2126R + 0.7152G + 0.0722B`).
3. The luminance is compared against the converted zone boundaries.
4. Zone edges are blended with a smoothstep S-curve for natural transitions.
5. Points inside a zone map to that zone's color; points outside pass through as grayscale.
6. The result is written as a standard `.cube` file.
