"""
False Color Exposure LUT Generator
====================================
Supports multiple camera log profiles.
Edit the CONFIG section — everything else is automatic.

Supported profiles:
  vlog      Panasonic V-Log L  (GH5, GH6, S5, S5II, BGH1...)
  slog3     Sony S-Log3        (A7S III, FX3, FX6, A7 IV, ZV-E1...)
  logc3     ARRI LogC3 EI800   (Alexa Mini, Alexa 35...)
  clog2     Canon C-Log2       (C70, C300 III, C500 II, EOS R5 C...)
  flog2     Fuji F-Log2        (X-H2S, X-H2...)
  nlog      Nikon N-Log        (Z6, Z7, Z6II, Z7II...)
  bmpfilm5  Blackmagic Film G5 (Pocket 6K G2, 6K Pro...)

Run with:
  uv run false_color_lut_generator.py
  uv run false_color_lut_generator.py --profile slog3 --size 65

Dependencies:
  colour-science  — vendor-accurate log transfer functions (https://www.colour-science.org/)
  numpy           — vectorised LUT generation
  pillow          — gradient preview PNG
  typer           — modern CLI with auto-complete and pretty --help
  rich            — coloured terminal output
"""

import os
from typing import Annotated, Optional

import warnings

import colour
import numpy as np
import typer

# colour-science's CanonLog2 v1.2 formula evaluates a branch for negative inputs
# as part of a vectorised where()-style implementation; the branch is never
# reached for valid scene-linear values but triggers a numpy RuntimeWarning.
warnings.filterwarnings("ignore", message="invalid value encountered in log10",
                        category=RuntimeWarning, module="colour")
from PIL import Image
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG — Only edit this section
# ═══════════════════════════════════════════════════════════════════════

# ── Which camera log profile are you grading? ──────────────────────────
# Options:  vlog | slog3 | logc3 | clog2 | flog2 | nlog | bmpfilm5
LOG_PROFILE = "vlog"

# ── LUT settings ───────────────────────────────────────────────────────
LUT_SIZE = 33  # 33 = standard,  65 = higher quality (4x larger file)

# ── Zone width and feathering ──────────────────────────────────────────
# Both values are in STOPS — converted to log units automatically per profile.
#
# KEY CONSTRAINT:  BLEND_WIDTH_STOPS / ZONE_HALF_WIDTH_STOPS  (the "blend ratio")
#
#   > 50%  Zone cores never reach full saturation — everything looks washed/mixed
#   ~ 40%  Cores reach solid color but blending eats most of the zone (too soft)
#   25–30% Sweet spot: solid core with a clean, smooth edge  ← aim for this
#   < 15%  Very hard edges; clinical/precise but can look harsh
#
#   Rule of thumb: keep the ratio at or below ⅓  (blend < half_width / 3)

ZONE_HALF_WIDTH_STOPS = 0.15  # Half-width of each color band (±stops from center)
#
# This controls how much of the tonal range is colored vs gray.
# Zones are spaced 1 stop apart, so the math is straightforward:
#
#   half_width   total zone width   gray gap between zones   % of range colored*
#   0.10         0.20 stops         0.80 stops               ~27%
#   0.15         0.30 stops         0.70 stops               ~33%  ← default
#   0.20         0.40 stops         0.60 stops               ~39%
#   0.30         0.60 stops         0.40 stops               ~51%  (too wide — mostly colored)
#   0.50         1.00 stops         0.00 stops               ~100% (zones touch, no gray at all)
#
#   * sampled uniformly across the full legal signal range

BLEND_WIDTH_STOPS = 0.04  # Feathering at zone edges (in stops)
# 0.03 = hard edge  |  0.10 = very soft
# Ratio with default half_width: 0.04/0.15 = 27%  ✓

# ── Zone definitions ───────────────────────────────────────────────────
# Format:  (stop_value_or_keyword,  "#HEXCOLOR")
#
# stop_value:
#   A number      →  stops relative to middle gray
#                    0 = mid gray,  +2 = two stops over,  -1 = one stop under
#   "white_clip"  →  top of the log curve (clipping highlights)
#   "black_clip"  →  bottom of the log curve (crushed blacks)
#
# Add, remove, or reorder rows freely.
# Any HTML hex color works — grab codes from Google Color Picker or coolors.co

ZONES = [
    ("white_clip", "#ef4444"),  # Clipping highlights → Red (red-500)
    (2, "#eab308"),  # +2 Stops            → Yellow (yellow-500)
    (1, "#d946ef"),  # +1 Stop             → Magenta (fuchsia-500)
    (0, "#22c55e"),  # Mid Gray (0 stops)  → Lime Green (green-500)
    (-1, "#06b6d4"),  # -1 Stop             → Cyan (cyan-500)
    (-2, "#3b82f6"),  # -2 Stops            → Royal Blue (blue-500)
    ("black_clip", "#6366f1"),  # Crushed blacks      → Indigo (indigo-500)
]

# ── Output filename ────────────────────────────────────────────────────
# {profile} is automatically replaced with your LOG_PROFILE value
OUTPUT_FILENAME = "luts/FalseColor_{profile}.cube"

# ═══════════════════════════════════════════════════════════════════════
#  END OF CONFIG — No need to edit below this line
# ═══════════════════════════════════════════════════════════════════════


console = Console()


# ───────────────────────────────────────────────────────────────────────
#  LOG PROFILE REGISTRY
#
#  Transfer functions are provided by colour-science
#  (https://www.colour-science.org/), which implements the published
#  vendor specifications for each log format.
#
#  linear_to_log : scene-linear (0.18 = 18% gray) → log-encoded (0.0–1.0)
#  log_to_linear : log-encoded (0.0–1.0) → scene-linear (exact inverse)
#
#  middle_gray   : reference log-encoded value of 18% gray (display only)
#  white_clip    : (lo, hi) log-encoded bounds for highlight clipping zone
#  black_clip    : (lo, hi) log-encoded bounds for shadow clipping zone
# ───────────────────────────────────────────────────────────────────────


def _logc3_encode(x):
    """ARRI LogC3 encode at EI800 — wrapper to lock in the exposure index."""
    return colour.models.log_encoding_ARRILogC3(x, EI=800)


def _logc3_decode(x):
    """ARRI LogC3 decode at EI800 — wrapper to lock in the exposure index."""
    return colour.models.log_decoding_ARRILogC3(x, EI=800)


PROFILES = {
    "vlog": {
        "name": "Panasonic V-Log L",
        "cameras": "GH5, GH6, S5, S5II, S1, BGH1, AU-EVA1",
        "linear_to_log": colour.models.log_encoding_VLog,
        "log_to_linear": colour.models.log_decoding_VLog,
        "middle_gray": 0.4233,  # log-encoded value of 18% gray (V-Log L spec)
        "white_clip": (0.80, 1.01),
        "black_clip": (0.00, 0.13),  # nominal black at 128/1024 = 0.125
    },
    "slog3": {
        "name": "Sony S-Log3",
        "cameras": "A7S III, A7 IV, FX3, FX6, FX9, ZV-E1, Venice",
        "linear_to_log": colour.models.log_encoding_SLog3,
        "log_to_linear": colour.models.log_decoding_SLog3,
        "middle_gray": 0.4106,  # 420/1023 per Sony S-Log3 spec
        "white_clip": (0.76, 1.00),
        "black_clip": (0.00, 0.10),  # nominal black at 95/1023 ≈ 0.0929
    },
    "logc3": {
        "name": "ARRI LogC3 (EI800)",
        "cameras": "Alexa Mini, Alexa Mini LF, Alexa 35, Amira",
        "linear_to_log": _logc3_encode,
        "log_to_linear": _logc3_decode,
        "middle_gray": 0.3910,
        "white_clip": (0.75, 1.00),
        "black_clip": (0.00, 0.10),  # nominal black at ~0.0928
    },
    "clog2": {
        "name": "Canon C-Log2",
        "cameras": "C70, C300 III, C500 II, EOS R5 C, EOS C70",
        "linear_to_log": colour.models.log_encoding_CanonLog2,
        "log_to_linear": colour.models.log_decoding_CanonLog2,
        "middle_gray": 0.3983,  # colour-science: log_encoding_CanonLog2(0.18)
        "white_clip": (0.70, 1.00),
        "black_clip": (0.00, 0.10),  # nominal black at ~0.0939 per colour-science
    },
    "flog2": {
        "name": "Fuji F-Log2",
        "cameras": "X-H2S, X-H2, GFX100S II",
        "linear_to_log": colour.models.log_encoding_FLog2,
        "log_to_linear": colour.models.log_decoding_FLog2,
        "middle_gray": 0.3910,  # colour-science: log_encoding_FLog2(0.18)
        "white_clip": (0.60, 1.00),
        "black_clip": (0.00, 0.10),  # nominal black at ~0.0937 per colour-science
    },
    "nlog": {
        "name": "Nikon N-Log",
        "cameras": "Z6, Z7, Z6II, Z7II, Z8, Z9",
        "linear_to_log": colour.models.log_encoding_NLog,
        "log_to_linear": colour.models.log_decoding_NLog,
        "middle_gray": 0.3637,  # colour-science: log_encoding_NLog(0.18)
        "white_clip": (0.72, 1.00),
        "black_clip": (0.00, 0.13),  # nominal black at ~0.125 per colour-science
    },
    "bmpfilm5": {
        "name": "Blackmagic Film Gen 5",
        "cameras": "Pocket 6K G2, 6K Pro, 6K G2, URSA Mini Pro 12K",
        "linear_to_log": colour.models.oetf_BlackmagicFilmGeneration5,
        "log_to_linear": colour.models.oetf_inverse_BlackmagicFilmGeneration5,
        "middle_gray": 0.3836,  # colour-science: oetf_BlackmagicFilmGeneration5(0.18)
        "white_clip": (0.75, 1.00),
        "black_clip": (0.00, 0.10),  # nominal black at ~0.0933 per colour-science
    },
}


# ───────────────────────────────────────────────────────────────────────
#  STOP → LOG CONVERSION
#  Because each log profile compresses stops differently, we use the
#  profile's own transfer function to calculate exact log-domain zone
#  boundaries. This means ±1 stop is always exactly ±1 stop of real
#  light, regardless of which profile you're using.
# ───────────────────────────────────────────────────────────────────────

MIDDLE_GRAY_LINEAR = 0.18  # Scene-linear value of 18% gray (industry standard)


def stop_to_log(stops: float, profile: dict) -> float:
    """
    Convert a stop offset (relative to middle gray) to a log-encoded value.
    Each stop doubles or halves the linear light level:
      +1 stop = 0.18 * 2^1 = 0.36 linear
      -2 stops = 0.18 * 2^-2 = 0.045 linear
    """
    linear = max(MIDDLE_GRAY_LINEAR * (2.0**stops), 1e-10)
    return float(profile["linear_to_log"](linear))


def stop_to_log_range(
    stops: float, half_width_stops: float, profile: dict
) -> tuple[float, float]:
    """
    Return (lo, hi) log boundaries for a zone centered at 'stops',
    spanning ±half_width_stops on either side.
    """
    lo = stop_to_log(stops - half_width_stops, profile)
    hi = stop_to_log(stops + half_width_stops, profile)
    return lo, hi


def blend_width_in_log(blend_width_stops: float, profile: dict) -> float:
    """
    Approximate blend width in log units, measured at middle gray.
    Used for informational display only; per-zone blend widths are
    computed more accurately in build_zones().
    """
    center = stop_to_log(0, profile)
    one_stop_up = stop_to_log(blend_width_stops, profile)
    return abs(one_stop_up - center)


# ───────────────────────────────────────────────────────────────────────
#  ZONE BUILDER
# ───────────────────────────────────────────────────────────────────────


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert '#FF0000' or 'FF0000' to normalized (r, g, b) floats 0.0–1.0."""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(
            f"Invalid hex color '#{hex_color}' — must be 6 digits, e.g. #FF0000"
        )
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def build_zones(
    zones_config: list,
    profile: dict,
    half_width_stops: float,
    blend_width_stops: float,
) -> list:
    """
    Resolve each entry in the user's ZONES list into a
    (lo, hi, r, g, b, bw) tuple with log-domain boundaries.

    The blend width (bw) is computed at each zone's own center stop so
    that feathering is accurate across the non-linear log curve — not just
    approximate at middle gray.  Clip zones fall back to a middle-gray
    approximation since they have no defined stop center.
    """
    parsed = []
    for stop_val, hex_color in zones_config:
        r, g, b = hex_to_rgb(hex_color)

        if stop_val == "white_clip":
            lo, hi = profile["white_clip"]
            center_stop = None
        elif stop_val == "black_clip":
            lo, hi = profile["black_clip"]
            center_stop = None
        elif isinstance(stop_val, (int, float)):
            lo, hi = stop_to_log_range(stop_val, half_width_stops, profile)
            center_stop = float(stop_val)
        else:
            raise ValueError(
                f"Unknown zone value '{stop_val}' — "
                f"use a number, 'white_clip', or 'black_clip'"
            )

        # Compute blend width in log units at this zone's center stop.
        if center_stop is not None:
            c = stop_to_log(center_stop, profile)
            up = stop_to_log(center_stop + blend_width_stops, profile)
        else:
            c = stop_to_log(0, profile)
            up = stop_to_log(blend_width_stops, profile)
        bw = abs(up - c)

        parsed.append((lo, hi, r, g, b, bw))

    return parsed


def check_zone_overlaps(parsed_zones: list) -> None:
    """Print a warning for any zones whose log ranges overlap."""
    sorted_z = sorted(parsed_zones, key=lambda z: z[0])
    for i in range(len(sorted_z) - 1):
        lo_a, hi_a = sorted_z[i][0], sorted_z[i][1]
        lo_b = sorted_z[i + 1][0]
        if hi_a > lo_b:
            console.print(
                f"  [yellow]⚠[/yellow]  Zone overlap: log {lo_a:.3f}–{hi_a:.3f} "
                f"overlaps log {lo_b:.3f}–{hi_a:.3f} "
                f"(shared range {lo_b:.3f}–{hi_a:.3f})"
            )


# ───────────────────────────────────────────────────────────────────────
#  CORE FALSE COLOR MATH
# ───────────────────────────────────────────────────────────────────────


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """S-curve interpolation: 0.0 at edge0, 1.0 at edge1, smooth in between."""
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def apply_false_color(
    r: float, g: float, b: float, parsed_zones: list, profile: dict
) -> tuple[float, float, float]:
    """
    Given log-encoded R,G,B (0.0–1.0):

    1. Decode each channel to scene-linear light via the profile's EOTF
    2. Compute Rec.709 luma (Y') in the *linear* domain — physically accurate
       because log curves compress stops non-uniformly, especially in shadows
    3. Re-encode that linear luma back to log for zone threshold comparisons
    4. Find the zone with the strongest match (with smooth edges)
    5. Return the blended false color, or grayscale if outside all zones

    Computing luma on linear light ensures one "stop" of exposure always
    spans the same fraction of the tonal scale, from deep shadows to highlights.
    """
    log_to_linear = profile["log_to_linear"]
    linear_to_log = profile["linear_to_log"]

    # Step 1 — decode log → linear per channel
    r_lin = float(log_to_linear(r))
    g_lin = float(log_to_linear(g))
    b_lin = float(log_to_linear(b))

    # Step 2 — Rec.709 luma in linear light
    luma_linear = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # Step 3 — re-encode luma to log for zone comparisons
    luma = float(linear_to_log(max(luma_linear, 1e-10)))

    best_weight = 0.0
    best_color = None

    for lo, hi, fr, fg, fb, bw in parsed_zones:
        fade_in = smoothstep(lo - bw, lo + bw, luma)
        fade_out = 1.0 - smoothstep(hi - bw, hi + bw, luma)
        weight = fade_in * fade_out

        if weight > best_weight:
            best_weight = weight
            best_color = (fr, fg, fb)

    if best_color is None or best_weight == 0.0:
        return luma, luma, luma  # grayscale passthrough

    fr, fg, fb = best_color
    return (
        fr * best_weight + luma * (1.0 - best_weight),
        fg * best_weight + luma * (1.0 - best_weight),
        fb * best_weight + luma * (1.0 - best_weight),
    )


# ───────────────────────────────────────────────────────────────────────
#  LUT BUILDER AND WRITER
# ───────────────────────────────────────────────────────────────────────


def build_lut(
    parsed_zones: list, size: int, profile: dict
) -> list[tuple[float, float, float]]:
    """
    Build the full size³ LUT table using NumPy vectorization.

    Luma is computed in the correct physical order:
      1. Decode log R/G/B → linear light  (colour-science transfer functions)
      2. Apply Rec.709 luma coefficients to linear values
      3. Re-encode linear luma → log for zone threshold comparisons

    This ensures one stop of exposure spans the same log-space fraction
    regardless of where it falls on the tonal scale (unlike applying
    Rec.709 coefficients directly to the log-encoded signal).

    Output order matches the .cube format: B (outer) → G → R (inner).
    """
    console.print(
        f"  Building [dim]{size}³[/dim] LUT [dim]({size**3:,} entries)[/dim]..."
    )

    vals = np.linspace(0.0, 1.0, size)

    # Shape: (size, size, size) indexed as [ri, gi, bi]
    R, G, B = np.meshgrid(vals, vals, vals, indexing="ij")

    log_to_linear = profile["log_to_linear"]
    linear_to_log = profile["linear_to_log"]

    # Step 1 — decode log → linear (colour-science handles numpy arrays natively)
    R_lin = np.asarray(log_to_linear(R), dtype=np.float64)
    G_lin = np.asarray(log_to_linear(G), dtype=np.float64)
    B_lin = np.asarray(log_to_linear(B), dtype=np.float64)

    # Step 2 — Rec.709 luma in linear light
    luma_linear = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin

    # Step 3 — re-encode luma to log for zone matching
    luma = np.asarray(linear_to_log(np.maximum(luma_linear, 1e-10)), dtype=np.float64)

    # Start with grayscale passthrough; overwrite where zones match
    out_r = luma.copy()
    out_g = luma.copy()
    out_b = luma.copy()
    best_weight = np.zeros_like(luma)

    for lo, hi, fr, fg, fb, bw in parsed_zones:
        if bw == 0.0:
            fade_in = np.where(luma >= lo, 1.0, 0.0)
            fade_out = np.where(luma < hi, 1.0, 0.0)
        else:
            t_in = np.clip((luma - (lo - bw)) / (2.0 * bw), 0.0, 1.0)
            fade_in = t_in * t_in * (3.0 - 2.0 * t_in)
            t_out = np.clip((luma - (hi - bw)) / (2.0 * bw), 0.0, 1.0)
            fade_out = 1.0 - t_out * t_out * (3.0 - 2.0 * t_out)

        weight = fade_in * fade_out
        mask = weight > best_weight

        mixed_r = fr * weight + luma * (1.0 - weight)
        mixed_g = fg * weight + luma * (1.0 - weight)
        mixed_b = fb * weight + luma * (1.0 - weight)

        best_weight = np.where(mask, weight, best_weight)
        out_r = np.where(mask, mixed_r, out_r)
        out_g = np.where(mask, mixed_g, out_g)
        out_b = np.where(mask, mixed_b, out_b)

    out_r = np.clip(out_r, 0.0, 1.0)
    out_g = np.clip(out_g, 0.0, 1.0)
    out_b = np.clip(out_b, 0.0, 1.0)

    # Reorder to .cube iteration: B (outer) → G → R (inner).
    # Arrays are indexed [ri, gi, bi]; transpose(2,1,0) → [bi, gi, ri] then ravel.
    flat_r = out_r.transpose(2, 1, 0).ravel()
    flat_g = out_g.transpose(2, 1, 0).ravel()
    flat_b = out_b.transpose(2, 1, 0).ravel()

    lut = list(zip(flat_r.tolist(), flat_g.tolist(), flat_b.tolist()))
    console.print(f"  [green]✓[/green] {len(lut):,} entries generated.")
    return lut


def write_cube(
    lut: list,
    size: int,
    filename: str,
    profile: dict,
    zones_config: list,
    parsed_zones: list,
) -> None:
    """Write the LUT table to a .cube file with a descriptive header."""
    zone_lines = []
    for (stop_val, hex_col), (lo, hi, *_) in zip(zones_config, parsed_zones):
        label = (
            "white_clip"
            if stop_val == "white_clip"
            else "black_clip"
            if stop_val == "black_clip"
            else f"{stop_val:+.0f} stops"
        )
        zone_lines.append(f"#   {hex_col:<10}  {label:<14}  log {lo:.3f} - {hi:.3f}")

    header = (
        f"# False Color Exposure Monitor LUT\n"
        f"# Profile  : {profile['name']}\n"
        f"# Cameras  : {profile['cameras']}\n"
        f"# Generator: false_color_lut_generator.py\n"
        f"#\n"
        f"# Zone Reference:\n"
        + "\n".join(zone_lines)
        + f"\n#   Grayscale   = between zones / untagged\n"
        f"#\n"
        f'TITLE "FalseColor {profile["name"]}"\n'
        f"LUT_3D_SIZE {size}\n"
        f"DOMAIN_MIN 0.0 0.0 0.0\n"
        f"DOMAIN_MAX 1.0 1.0 1.0\n\n"
    )

    with open(filename, "w") as f:
        f.write(header)
        for r, g, b in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    console.print(f"  [green]✓[/green] Written to: [bold]{filename}[/bold]")


# ───────────────────────────────────────────────────────────────────────
#  GRADIENT PREVIEW IMAGE  (Pillow)
# ───────────────────────────────────────────────────────────────────────


def generate_gradient_preview(
    parsed_zones: list,
    profile: dict,
    output_path: str,
    width: int = 1920,
    height: int = 200,
) -> None:
    """
    Render a horizontal gradient (log 0→1) with false color applied and
    save it as a PNG alongside the .cube file.

    The image gives an instant visual reference of the zone layout —
    which tonal ranges are colored, and what colors they map to —
    without needing to open Resolve or Premiere.

    For a neutral gray ramp (R=G=B), the linear luma equals the decoded
    linear value, so re-encoding gives back the original log value.
    The false color zones appear exactly where configured.
    """
    t = np.linspace(0.0, 1.0, width)

    # Apply false color to each column of the gradient
    row = np.zeros((width, 3), dtype=np.uint8)
    for i, v in enumerate(t):
        r_out, g_out, b_out = apply_false_color(
            float(v), float(v), float(v), parsed_zones, profile
        )
        row[i] = [
            int(max(0.0, min(1.0, r_out)) * 255),
            int(max(0.0, min(1.0, g_out)) * 255),
            int(max(0.0, min(1.0, b_out)) * 255),
        ]

    img_array = np.tile(row, (height, 1, 1))
    Image.fromarray(img_array, mode="RGB").save(output_path)
    console.print(f"  [green]✓[/green] Preview saved: [bold]{output_path}[/bold]")


# ───────────────────────────────────────────────────────────────────────
#  ZONE PREVIEW TABLE  (Rich)
# ───────────────────────────────────────────────────────────────────────


def _swatch(hex_col: str) -> str:
    """
    Rich markup for a color swatch: colored background block with the hex code.
    Text color (black/white) is chosen automatically for contrast.
    """
    r = int(hex_col[1:3], 16) / 255.0
    g = int(hex_col[3:5], 16) / 255.0
    b = int(hex_col[5:7], 16) / 255.0
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fg = "black" if luma > 0.45 else "white"
    return f"[{fg} on {hex_col}] {hex_col} [/]"


def print_zone_table(zones_config: list, parsed_zones: list, profile: dict) -> None:
    """Print a Rich table showing each zone with live terminal color swatches."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Zone", style="bold", min_width=16)
    table.add_column("Log Range", style="dim", min_width=15)
    table.add_column("Target", justify="center", min_width=14)
    table.add_column("Output", justify="center", min_width=14)
    table.add_column("Match", justify="center", min_width=5)

    for (stop_val, hex_col), (lo, hi, fr, fg, fb, bw) in zip(
        zones_config, parsed_zones
    ):
        label = (
            "white_clip"
            if stop_val == "white_clip"
            else "black_clip"
            if stop_val == "black_clip"
            else f"{stop_val:+.0f} stops"
        )
        center = (lo + hi) / 2.0
        r_out, g_out, b_out = apply_false_color(
            center, center, center, parsed_zones, profile
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        match = (
            "[green]✓[/green]"
            if hex_out.upper() == hex_col.upper()
            else "[yellow]~[/yellow]"
        )
        table.add_row(
            label, f"{lo:.3f} – {hi:.3f}", _swatch(hex_col), _swatch(hex_out), match
        )

    # Verify a between-zone point (should be gray)
    if len(parsed_zones) >= 2:
        sorted_z = sorted(parsed_zones, key=lambda z: z[0])
        gap_mid = (sorted_z[0][1] + sorted_z[1][0]) / 2.0
        r_out, g_out, b_out = apply_false_color(
            gap_mid, gap_mid, gap_mid, parsed_zones, profile
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        table.add_row(
            "[dim](between zones)[/dim]",
            f"[dim]{gap_mid:.3f}[/dim]",
            "[dim](gray)[/dim]",
            _swatch(hex_out),
            "–",
        )

    console.print(table)


# ───────────────────────────────────────────────────────────────────────
#  CLI  (Typer)
# ───────────────────────────────────────────────────────────────────────

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Generate a false color exposure LUT for a camera log profile.",
)


@app.command()
def main(
    profile: Annotated[
        str,
        typer.Option(
            help=f"Camera log profile. Options: {', '.join(PROFILES.keys())}",
            show_default=True,
        ),
    ] = LOG_PROFILE,
    size: Annotated[
        int,
        typer.Option(
            metavar="N",
            help="LUT grid size. Standard NLE values: 17, 33, 65.",
        ),
    ] = LUT_SIZE,
    half_width: Annotated[
        float,
        typer.Option(
            "--half-width",
            metavar="STOPS",
            help="Zone half-width in stops (±stops from zone center).",
        ),
    ] = ZONE_HALF_WIDTH_STOPS,
    blend: Annotated[
        float,
        typer.Option(
            metavar="STOPS",
            help="Feathering width at zone edges in stops.",
        ),
    ] = BLEND_WIDTH_STOPS,
    output: Annotated[
        Optional[str],
        typer.Option(
            metavar="FILE",
            help="Output .cube filename. Default: luts/FalseColor_{profile}.cube",
        ),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(
            help="Generate a gradient preview PNG alongside the .cube file.",
        ),
    ] = True,
) -> None:
    """Generate a false color exposure LUT for a camera log profile."""

    # ── Validate inputs ───────────────────────────────────────────────
    if profile not in PROFILES:
        console.print(
            f"[bold red]Error:[/bold red] Unknown profile [bold]'{profile}'[/bold]"
        )
        console.print(f"[dim]Valid options: {', '.join(PROFILES.keys())}[/dim]")
        raise typer.Exit(1)

    if size < 2:
        console.print(
            f"[bold red]Error:[/bold red] --size must be at least 2, got {size}"
        )
        raise typer.Exit(1)

    log_profile_obj = PROFILES[profile]
    filename = (output or OUTPUT_FILENAME).replace("{profile}", profile)
    blend_log = blend_width_in_log(blend, log_profile_obj)

    # ── Header ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]False Color Exposure LUT Generator[/bold]", style="dim"))
    console.print()
    console.print(f"  [dim]Profile   :[/dim]  {log_profile_obj['name']}")
    console.print(f"  [dim]Cameras   :[/dim]  [dim]{log_profile_obj['cameras']}[/dim]")
    console.print(
        f"  [dim]Mid gray  :[/dim]  {log_profile_obj['middle_gray']:.4f}  [dim](log-encoded 18% gray)[/dim]"
    )
    console.print(f"  [dim]Zones     :[/dim]  {len(ZONES)}")
    console.print(f"  [dim]Zone width:[/dim]  ±{half_width} stops")
    console.print(
        f"  [dim]Blend     :[/dim]  {blend} stops  [dim]({blend_log:.4f} log units at mid-gray)[/dim]"
    )
    console.print(
        f"  [dim]LUT size  :[/dim]  {size}³  [dim]({size**3:,} entries)[/dim]"
    )
    console.print(f"  [dim]Output    :[/dim]  {filename}")

    if size not in (17, 33, 65):
        console.print(
            f"\n  [yellow]⚠[/yellow]  Non-standard LUT size {size}. Common NLE sizes are 17, 33, 65."
        )

    # ── Blend ratio analysis ──────────────────────────────────────────
    blend_ratio = blend / half_width
    solid_core_stops = 2.0 * (half_width - blend)

    if blend_ratio > 0.50:
        ratio_label = (
            "[red]POOR[/red]   — zones never reach solid color (cores overlap)"
        )
    elif blend_ratio > 0.40:
        ratio_label = "[yellow]SOFT[/yellow]   — cores barely solid, blending dominates"
    elif blend_ratio > 0.33:
        ratio_label = "[dim]OK[/dim]     — usable but softer than ideal"
    elif blend_ratio >= 0.15:
        ratio_label = "[green]GOOD[/green]   — solid core with smooth edges"
    else:
        ratio_label = "[dim]SHARP[/dim]  — hard edges, clinical look"

    console.print(f"\n  [dim]Blend ratio:[/dim]  {blend_ratio:.0%}  {ratio_label}")
    console.print(
        f"  [dim]Solid core:[/dim]   {solid_core_stops:.3f} stops  "
        f"[dim](zone is {half_width * 2:.2f} stops wide, "
        f"{blend_ratio * 100:.0f}% consumed by fade ramps)[/dim]"
    )
    if blend_ratio > 0.40:
        console.print(
            f"\n  [yellow]⚠[/yellow]  Consider reducing [bold]--blend[/bold] to "
            f"[bold]{half_width * 0.27:.2f}[/bold] "
            f"[dim](27% of half-width = solid-core sweet spot)[/dim]"
        )

    # ── Zone preview table ────────────────────────────────────────────
    console.print()
    parsed_zones = build_zones(ZONES, log_profile_obj, half_width, blend)
    check_zone_overlaps(parsed_zones)
    print_zone_table(ZONES, parsed_zones, log_profile_obj)

    # ── Build and write ───────────────────────────────────────────────
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    lut_data = build_lut(parsed_zones, size, log_profile_obj)
    write_cube(lut_data, size, filename, log_profile_obj, ZONES, parsed_zones)

    if preview:
        preview_path = os.path.splitext(filename)[0] + "_preview.png"
        generate_gradient_preview(parsed_zones, log_profile_obj, preview_path)

    # ── Footer ────────────────────────────────────────────────────────
    size_kb = os.path.getsize(filename) / 1024
    console.print()
    console.print(
        f"[bold green]✅ Done![/bold green]  [bold]{filename}[/bold]  [dim]({size_kb:.1f} KB)[/dim]"
    )
    console.print(
        "   Drop it into [bold]DaVinci Resolve[/bold], [bold]Premiere Pro[/bold], "
        "[bold]FCPX[/bold], or your camera's LUT slot."
    )
    console.print(Rule(style="dim"))
    console.print()


if __name__ == "__main__":
    app()
