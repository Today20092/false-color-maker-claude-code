# V-Log False Color LUT Generator

A Python script that generates `.cube` LUT files for false color exposure monitoring. Designed for Panasonic V-Log (Lumix S5IIX and compatible cameras), but works with any V-Log footage in post.

The LUT maps specific luminance zones to distinct colors so you can instantly spot underexposed and overexposed areas in your footage.

## Default Zone Map

| Zone | V-Log Range | Color |
|------|-------------|-------|
| White Clip | 0.80 – 1.01 | Red `#ef4444` |
| +2 Stops | 0.56 – 0.60 | Yellow `#fde047` |
| +1 Stop | 0.48 – 0.52 | Magenta `#d946ef` |
| Mid Gray | 0.40 – 0.44 | Green `#22c55e` |
| -1 Stop | 0.32 – 0.36 | Cyan `#06b6d4` |
| -2 Stops | 0.24 – 0.28 | Blue `#3b82f6` |
| Black Clip | 0.00 – 0.10 | Dark Purple `#6d28d9` |
| Between zones | — | Grayscale passthrough |

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

```bash
python vlog_false_color_generator.py
```

This prints a zone preview to the terminal, generates the LUT, and writes the `.cube` file to disk.

## Configuration

All settings are at the top of `vlog_false_color_generator.py` in the `CONFIG` section. You do not need to touch anything else.

### LUT Size

```python
LUT_SIZE = 33   # 33 = standard, 65 = higher quality (larger file)
```

### Blend Width

Controls how softly zone edges transition into grayscale.

```python
BLEND_WIDTH = 0.009   # 0.005 = tight  |  0.015 = soft  |  0.030 = wide/dreamy
```

### Output Filename

```python
OUTPUT_FILENAME = "VLog_FalseColor_Exposure.cube"
```

### Custom Zones

Edit the `ZONES` list. Each row is `(v_log_min, v_log_max, "hex_color")`.

```python
ZONES = [
    (0.80, 1.01, "#ef4444"),   # White Clip
    (0.40, 0.44, "#22c55e"),   # Mid Gray
    (0.00, 0.10, "#6d28d9"),   # Black Clip
]
```

Anything outside all defined zones renders as grayscale.

## Using the Output LUT

Drop the generated `.cube` file into:

- **DaVinci Resolve** — Color page > LUTs > Apply as a node
- **Premiere Pro** — Lumetri Color > Creative > Look
- **Lumix Camera** — Load into a LUT slot via the camera menu (check your camera's manual for the correct format/size)

## How It Works

1. For each point in the 3D color cube, the script calculates perceptual luminance using Rec.709 weights (`0.2126R + 0.7152G + 0.0722B`).
2. It checks which zone that luminance value falls into.
3. Zone edges are blended using a smoothstep S-curve for natural transitions.
4. Points inside a zone are mapped to the zone's color; points outside are passed through as grayscale.
5. The result is written as a standard `.cube` file.
