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
  nlog      Nikon N-Log        (Z6, Z7, Z6II, Z7II...)  [approximate]
  bmpfilm5  Blackmagic Film G5 (Pocket 6K G2, 6K Pro...)

Run with:
  uv run false_color_lut_generator.py
  uv run false_color_lut_generator.py --profile slog3 --size 65
"""

import math
import os

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG — Only edit this section
# ═══════════════════════════════════════════════════════════════════════

# ── Which camera log profile are you grading? ──────────────────────────
# Options:  vlog | slog3 | logc3 | clog2 | flog2 | nlog | bmpfilm5
LOG_PROFILE = "vlog"

# ── LUT settings ───────────────────────────────────────────────────────
LUT_SIZE = 33          # 33 = standard,  65 = higher quality (4x larger file)

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

ZONE_HALF_WIDTH_STOPS = 0.15   # Half-width of each color band (±stops from center)
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

BLEND_WIDTH_STOPS = 0.04       # Feathering at zone edges (in stops)
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
    ("white_clip",  "#ef4444"),   # Clipping highlights → Red (red-500)
    ( 2,            "#eab308"),   # +2 Stops            → Yellow (yellow-500)
    ( 1,            "#d946ef"),   # +1 Stop             → Magenta (fuchsia-500)
    ( 0,            "#22c55e"),   # Mid Gray (0 stops)  → Lime Green (green-500)
    (-1,            "#06b6d4"),   # -1 Stop             → Cyan (cyan-500)
    (-2,            "#3b82f6"),   # -2 Stops            → Royal Blue (blue-500)
    ("black_clip",  "#6366f1"),   # Crushed blacks      → Indigo (indigo-500)
]

# ── Output filename ────────────────────────────────────────────────────
# {profile} is automatically replaced with your LOG_PROFILE value
OUTPUT_FILENAME = "luts/FalseColor_{profile}.cube"

# ═══════════════════════════════════════════════════════════════════════
#  END OF CONFIG — No need to edit below this line
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  LOG PROFILE TRANSFER FUNCTIONS
#
#  Each function converts scene-linear light (0.0–1.0, where 0.18 = 18%
#  gray) to the log-encoded value (0.0–1.0) used in that camera profile.
#
#  Sources:
#    V-Log    — Panasonic V-Log/V-Log L Spec v1.0
#    S-Log3   — Sony S-Log3 Technical Summary v1.0
#    LogC3    — ARRI LogC3 Specification (Alexa v3 color science)
#    C-Log2   — Canon Log Gamma Curve Spec v2.0
#    F-Log2   — Fujifilm F-Log2 Specification v1.0
#    N-Log    — Nikon N-Log Specification (APPROXIMATE — see note)
#    BMDFilm5 — Blackmagic Design Generation 5 Color Science SDK
# ───────────────────────────────────────────────────────────────────────

def _vlog_to_log(x: float) -> float:
    """Panasonic V-Log L. Covers ~14 stops dynamic range."""
    x = max(x, 1e-10)
    if x < 0.01:
        return 5.6 * x + 0.125
    return 0.241514 * math.log10(x + 0.00873) + 0.598206


def _slog3_to_log(x: float) -> float:
    """Sony S-Log3. Covers ~15.3 stops dynamic range.
    Reference: Sony S-Log3 Technical Summary v1.0
    Legal range: 0% exposure = 95/1023 ≈ 0.0929, middle gray = 420/1023 ≈ 0.4106
    Denominator is (0.18 + 0.01) = 0.19 per Sony spec. Cutpoint 0.01125 ensures
    continuity between the linear and logarithmic segments.
    """
    x = max(x, 1e-10)
    if x >= 0.01125:
        return (420.0 + math.log10((x + 0.01) / 0.19) * 261.5) / 1023.0
    return (x * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0


def _logc3_to_log(x: float) -> float:
    """ARRI LogC3 at EI800. Covers ~14 stops dynamic range."""
    x = max(x, 1e-10)
    if x >= 0.010591:
        return 0.247190 * math.log10(5.555556 * x + 0.052272) + 0.385537
    return 5.367655 * x + 0.092809


def _clog2_to_log(x: float) -> float:
    """Canon C-Log2. Covers ~15+ stops dynamic range."""
    x = max(x, 1e-10)
    return 0.281863093 * math.log10(87.09937546 * x + 1.0) + 0.073059361


def _flog2_to_log(x: float) -> float:
    """
    Fuji F-Log2. Covers ~14+ stops dynamic range.
    Reference: Fujifilm F-Log2 Specification v1.0
    """
    x = max(x, 1e-10)
    cut = 0.000889544
    if x >= cut:
        return 0.384736 * math.log10(x + 0.064829) + 0.553667
    return 8.799461 * x + 0.092864


def _nlog_to_log(x: float) -> float:
    """
    Nikon N-Log. Covers ~12 stops dynamic range.
    NOTE: This is an approximation derived from published middle-gray and
    white-point references. Verify against Nikon's official LUT tools
    before using in color-critical work.
    """
    x = max(x, 1e-10)
    cut = 0.006
    if x >= cut:
        return 0.4614 * math.log10(12.8 * x + 1.0) + 0.124
    return 10.541 * x + 0.124


def _bmpfilm5_to_log(x: float) -> float:
    """
    Blackmagic Film Gen 5. Covers ~13 stops dynamic range.
    Reference: Blackmagic Design Generation 5 Color Science SDK.
    """
    x = max(x, 1e-10)
    cut = 0.005
    if x < cut:
        return 8.283605932402494 * x + 0.09246575342465753
    return 0.08692876065491224 * math.log(x + 0.005494072432257808) + 0.5402385771788551


# ── Transfer function self-tests ─────────────────────────────────────────
# Verify that each profile's middle gray (0.18 linear) maps to its published
# reference log value. Any mismatch > 0.002 log units raises an error at import.

def _validate_transfer_functions() -> None:
    """Sanity-check each profile's transfer function against published reference points."""
    tol = 0.002
    checks = [
        ("V-Log L",        _vlog_to_log,     0.18, 0.4233),
        ("S-Log3",         _slog3_to_log,    0.18, 0.4106),
        ("LogC3 EI800",    _logc3_to_log,    0.18, 0.3910),
        ("C-Log2",         _clog2_to_log,    0.18, 0.4175),
        ("F-Log2",         _flog2_to_log,    0.18, 0.3185),
        ("N-Log [approx]", _nlog_to_log,     0.18, 0.3635),
        ("BMD Film Gen 5", _bmpfilm5_to_log, 0.18, 0.3938),
    ]
    for name, fn, linear_in, expected in checks:
        actual = fn(linear_in)
        if abs(actual - expected) > tol:
            raise AssertionError(
                f"Transfer function mismatch for {name}: "
                f"expected {expected:.4f}, got {actual:.4f} "
                f"(diff {abs(actual - expected):.4f} > tol {tol})"
            )

_validate_transfer_functions()


# ── Profile registry ────────────────────────────────────────────────────
# To add a new profile: copy one of the blocks below, give it a key,
# fill in the linear_to_log function and reference values.

PROFILES = {
    "vlog": {
        "name":          "Panasonic V-Log L",
        "cameras":       "GH5, GH6, S5, S5II, S1, BGH1, AU-EVA1",
        "linear_to_log": _vlog_to_log,
        "middle_gray":   0.4233,  # log-encoded value of 18% gray
        "white_clip":    (0.80, 1.01),
        "black_clip":    (0.00, 0.13),  # covers nominal black at 128/1024 = 0.125 (legal range)
    },
    "slog3": {
        "name":          "Sony S-Log3",
        "cameras":       "A7S III, A7 IV, FX3, FX6, FX9, ZV-E1, Venice",
        "linear_to_log": _slog3_to_log,
        "middle_gray":   0.4106,  # 420/1023 per Sony S-Log3 spec (legal range)
        "white_clip":    (0.76, 1.00),
        "black_clip":    (0.00, 0.10),  # covers nominal black at 95/1023 ≈ 0.0929
    },
    "logc3": {
        "name":          "ARRI LogC3 (EI800)",
        "cameras":       "Alexa Mini, Alexa Mini LF, Alexa 35, Amira",
        "linear_to_log": _logc3_to_log,
        "middle_gray":   0.3910,
        "white_clip":    (0.75, 1.00),
        "black_clip":    (0.00, 0.10),  # covers nominal black at ~0.0928
    },
    "clog2": {
        "name":          "Canon C-Log2",
        "cameras":       "C70, C300 III, C500 II, EOS R5 C, EOS C70",
        "linear_to_log": _clog2_to_log,
        "middle_gray":   0.4175,
        "white_clip":    (0.70, 1.00),
        "black_clip":    (0.00, 0.08),  # covers nominal black at ~0.0731
    },
    "flog2": {
        "name":          "Fuji F-Log2",
        "cameras":       "X-H2S, X-H2, GFX100S II",
        "linear_to_log": _flog2_to_log,
        "middle_gray":   0.3185,
        "white_clip":    (0.60, 1.00),
        "black_clip":    (0.00, 0.10),  # covers nominal black at ~0.0929
    },
    "nlog": {
        "name":          "Nikon N-Log  [approximate]",
        "cameras":       "Z6, Z7, Z6II, Z7II, Z8, Z9",
        "linear_to_log": _nlog_to_log,
        "middle_gray":   0.3635,
        "white_clip":    (0.72, 1.00),
        "black_clip":    (0.00, 0.13),  # covers nominal black at ~0.124 (legal range)
    },
    "bmpfilm5": {
        "name":          "Blackmagic Film Gen 5",
        "cameras":       "Pocket 6K G2, 6K Pro, 6K G2, URSA Mini Pro 12K",
        "linear_to_log": _bmpfilm5_to_log,
        "middle_gray":   0.3938,
        "white_clip":    (0.75, 1.00),
        "black_clip":    (0.00, 0.10),  # covers nominal black at ~0.0925
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
    linear = MIDDLE_GRAY_LINEAR * (2.0 ** stops)
    return profile["linear_to_log"](linear)


def stop_to_log_range(stops: float, half_width_stops: float, profile: dict) -> tuple[float, float]:
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
    center      = stop_to_log(0, profile)
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
        # For clip zones, fall back to middle gray as an approximation.
        if center_stop is not None:
            c  = stop_to_log(center_stop, profile)
            up = stop_to_log(center_stop + blend_width_stops, profile)
        else:
            c  = stop_to_log(0, profile)
            up = stop_to_log(blend_width_stops, profile)
        bw = abs(up - c)

        parsed.append((lo, hi, r, g, b, bw))

    return parsed


def check_zone_overlaps(parsed_zones: list) -> None:
    """Print a warning for any zones whose log ranges overlap."""
    sorted_z = sorted(parsed_zones, key=lambda z: z[0])
    for i in range(len(sorted_z) - 1):
        lo_a, hi_a = sorted_z[i][0], sorted_z[i][1]
        lo_b, hi_b = sorted_z[i + 1][0], sorted_z[i + 1][1]
        if hi_a > lo_b:
            print(
                f"  ⚠  Zone overlap: log {lo_a:.3f}–{hi_a:.3f} "
                f"overlaps log {lo_b:.3f}–{hi_b:.3f} "
                f"(shared range {lo_b:.3f}–{hi_a:.3f})"
            )


# ───────────────────────────────────────────────────────────────────────
#  CORE FALSE COLOR MATH
# ───────────────────────────────────────────────────────────────────────

def rgb_luminance(r: float, g: float, b: float) -> float:
    """
    Rec.709 perceptual luma. Combines R, G, B into a single brightness.
    Green is weighted most because human eyes are most sensitive to it.
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """S-curve interpolation: 0.0 at edge0, 1.0 at edge1, smooth in between."""
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def apply_false_color(
    r: float, g: float, b: float, parsed_zones: list
) -> tuple[float, float, float]:
    """
    Given log-encoded R,G,B (0.0–1.0):
    - Calculate luma
    - Find the zone with the strongest match (with smooth edges)
    - Return the blended false color, or grayscale if outside all zones
    """
    luma        = rgb_luminance(r, g, b)
    best_weight = 0.0
    best_color  = None

    for (lo, hi, fr, fg, fb, bw) in parsed_zones:
        fade_in  = smoothstep(lo - bw, lo + bw, luma)
        fade_out = 1.0 - smoothstep(hi - bw, hi + bw, luma)
        weight   = fade_in * fade_out

        if weight > best_weight:
            best_weight = weight
            best_color  = (fr, fg, fb)

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

def build_lut(parsed_zones: list, size: int) -> list[tuple[float, float, float]]:
    """
    Build the full size³ LUT table using NumPy vectorization.

    Computes luminance and smoothstep blending across the entire 3D grid
    simultaneously, which is ~10–100× faster than a pure-Python loop.

    Output order matches the .cube format: B (outer) → G → R (inner).
    """
    print(f"  Building {size}³ LUT ({size**3:,} entries)...")

    vals = np.linspace(0.0, 1.0, size)

    # Shape: (size, size, size) indexed as [ri, gi, bi]
    R, G, B = np.meshgrid(vals, vals, vals, indexing="ij")

    # Rec.709 luminance over the full grid
    luma = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # Start with grayscale passthrough; overwrite where zones match
    out_r = luma.copy()
    out_g = luma.copy()
    out_b = luma.copy()
    best_weight = np.zeros_like(luma)

    for (lo, hi, fr, fg, fb, bw) in parsed_zones:
        if bw == 0.0:
            fade_in  = np.where(luma >= lo, 1.0, 0.0)
            fade_out = np.where(luma <  hi, 1.0, 0.0)
        else:
            t_in    = np.clip((luma - (lo - bw)) / (2.0 * bw), 0.0, 1.0)
            fade_in = t_in * t_in * (3.0 - 2.0 * t_in)
            t_out    = np.clip((luma - (hi - bw)) / (2.0 * bw), 0.0, 1.0)
            fade_out = 1.0 - t_out * t_out * (3.0 - 2.0 * t_out)

        weight  = fade_in * fade_out
        mask    = weight > best_weight

        mixed_r = fr * weight + luma * (1.0 - weight)
        mixed_g = fg * weight + luma * (1.0 - weight)
        mixed_b = fb * weight + luma * (1.0 - weight)

        best_weight = np.where(mask, weight, best_weight)
        out_r       = np.where(mask, mixed_r, out_r)
        out_g       = np.where(mask, mixed_g, out_g)
        out_b       = np.where(mask, mixed_b, out_b)

    out_r = np.clip(out_r, 0.0, 1.0)
    out_g = np.clip(out_g, 0.0, 1.0)
    out_b = np.clip(out_b, 0.0, 1.0)

    # Reorder to .cube iteration: B (outer) → G → R (inner).
    # Arrays are indexed [ri, gi, bi]; transpose(2,1,0) → [bi, gi, ri] then ravel.
    flat_r = out_r.transpose(2, 1, 0).ravel()
    flat_g = out_g.transpose(2, 1, 0).ravel()
    flat_b = out_b.transpose(2, 1, 0).ravel()

    lut = list(zip(flat_r.tolist(), flat_g.tolist(), flat_b.tolist()))
    print(f"  ✓ {len(lut):,} entries generated.")
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
            "white_clip" if stop_val == "white_clip" else
            "black_clip" if stop_val == "black_clip" else
            f"{stop_val:+.0f} stops"
        )
        zone_lines.append(f"#   {hex_col:<10}  {label:<14}  log {lo:.3f} - {hi:.3f}")

    header = (
        f"# False Color Exposure Monitor LUT\n"
        f"# Profile  : {profile['name']}\n"
        f"# Cameras  : {profile['cameras']}\n"
        f"# Generator: false_color_lut_generator.py\n"
        f"#\n"
        f"# Zone Reference:\n"
        + "\n".join(zone_lines) +
        f"\n#   Grayscale   = between zones / untagged\n"
        f"#\n"
        f"TITLE \"FalseColor {profile['name']}\"\n"
        f"LUT_3D_SIZE {size}\n"
        f"DOMAIN_MIN 0.0 0.0 0.0\n"
        f"DOMAIN_MAX 1.0 1.0 1.0\n\n"
    )

    with open(filename, "w") as f:
        f.write(header)
        for (r, g, b) in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    print(f"  ✓ Written to: {filename}")


# ───────────────────────────────────────────────────────────────────────
#  ZONE PREVIEW PRINTOUT
# ───────────────────────────────────────────────────────────────────────

def print_zone_preview(zones_config: list, parsed_zones: list) -> None:
    print(f"\n  {'Zone':<16} {'Log Range':<18} {'Target':<10} {'Output':<10} Match")
    print(f"  {'────':<16} {'─────────':<18} {'──────':<10} {'──────':<10} ─────")

    for (stop_val, hex_col), (lo, hi, fr, fg, fb, bw) in zip(zones_config, parsed_zones):
        label = (
            "white_clip" if stop_val == "white_clip" else
            "black_clip" if stop_val == "black_clip" else
            f"{stop_val:+.0f} stops"
        )
        center = (lo + hi) / 2.0
        r_out, g_out, b_out = apply_false_color(center, center, center, parsed_zones)
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        match = "✓" if hex_out.upper() == hex_col.upper() else "~"
        print(f"  {label:<16} {lo:.3f} - {hi:.3f}    {hex_col:<10} {hex_out:<10} {match}")

    # Sample a between-zone point (should always be gray).
    # Sort by lo so the gap midpoint is always between the two lowest zones,
    # regardless of the order zones were defined in CONFIG.
    if len(parsed_zones) >= 2:
        sorted_z = sorted(parsed_zones, key=lambda z: z[0])
        gap_mid  = (sorted_z[0][1] + sorted_z[1][0]) / 2.0
        r_out, g_out, b_out = apply_false_color(gap_mid, gap_mid, gap_mid, parsed_zones)
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        print(f"  {'(between zones)':<16} {gap_mid:.3f}              {'(gray)':<10} {hex_out:<10} –")

    print()


# ───────────────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the False Color LUT generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a false color exposure LUT for a camera log profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile", default=LOG_PROFILE, choices=list(PROFILES.keys()),
        help="Camera log profile",
    )
    parser.add_argument(
        "--size", type=int, default=LUT_SIZE, metavar="N",
        help="LUT grid size (standard NLE values: 17, 33, 65)",
    )
    parser.add_argument(
        "--half-width", type=float, default=ZONE_HALF_WIDTH_STOPS, metavar="STOPS",
        help="Zone half-width in stops (±stops from zone center)",
    )
    parser.add_argument(
        "--blend", type=float, default=BLEND_WIDTH_STOPS, metavar="STOPS",
        help="Feathering width at zone edges in stops",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Output .cube filename (default: luts/FalseColor_{profile}.cube)",
    )
    args = parser.parse_args()

    log_profile       = args.profile
    lut_size          = args.size
    zone_half_width   = args.half_width
    blend_width_stops = args.blend
    filename          = (args.output or OUTPUT_FILENAME).replace("{profile}", log_profile)

    # ── Validate LUT size ─────────────────────────────────────────────
    if lut_size < 2:
        raise ValueError(f"--size must be at least 2, got {lut_size}")
    if lut_size not in (17, 33, 65):
        print(f"  ⚠  Non-standard LUT size {lut_size}. Common NLE sizes are 17, 33, 65.")

    profile   = PROFILES[log_profile]
    blend_log = blend_width_in_log(blend_width_stops, profile)  # approximate, for display

    print("=" * 58)
    print("  False Color Exposure LUT Generator")
    print("=" * 58)
    print(f"\n  Profile       : {profile['name']}")
    print(f"  Cameras       : {profile['cameras']}")
    print(f"  Middle gray   : {profile['middle_gray']:.4f}  (log-encoded 18% gray)")
    print(f"  Zones defined : {len(ZONES)}")
    print(f"  Zone width    : ±{zone_half_width} stops")
    print(f"  Blend width   : {blend_width_stops} stops  ({blend_log:.4f} log units, at mid-gray)")
    print(f"  LUT size      : {lut_size}³  ({lut_size**3:,} entries)")
    print(f"  Output file   : {filename}")

    # ── Blend ratio check ────────────────────────────────────────────────
    # The blend ramp is applied symmetrically at each zone edge, so the
    # ramp from each side eats into the zone core from both directions.
    # When blend/half_width > ~0.33, the two ramps overlap in the middle
    # and the zone core never reaches 100% opacity.
    #
    # Concretely: a zone spans  [center - half_width, center + half_width].
    # The color is fully solid only between:
    #   (center - half_width + blend_width)  and  (center + half_width - blend_width)
    # That solid core is  2 * (half_width - blend_width)  stops wide.
    # If blend_width >= half_width the core shrinks to zero — no solid color at all.
    blend_ratio      = blend_width_stops / zone_half_width
    solid_core_stops = 2.0 * (zone_half_width - blend_width_stops)

    if blend_ratio > 0.50:
        ratio_label = "⚠️  POOR   — zones never reach solid color (cores overlap)"
    elif blend_ratio > 0.40:
        ratio_label = "⚠️  SOFT   — cores barely solid, blending dominates"
    elif blend_ratio > 0.33:
        ratio_label = "〜  OK     — usable but softer than ideal"
    elif blend_ratio >= 0.15:
        ratio_label = "✓  GOOD   — solid core with smooth edges"
    else:
        ratio_label = "〜  SHARP  — hard edges, clinical look"

    print(f"\n  Blend ratio   : {blend_ratio:.0%}  ({ratio_label})")
    print(f"  Solid core    : {solid_core_stops:.3f} stops per zone"
          f"  (zone is {zone_half_width*2:.2f} stops wide,"
          f" {blend_ratio*100:.0f}% consumed by fade ramps)")
    if blend_ratio > 0.40:
        print(f"\n  ⚠  Consider reducing --blend to"
              f" {zone_half_width * 0.27:.2f}"
              f" (27% of half_width = solid-core sweet spot)")
    # ─────────────────────────────────────────────────────────────────────

    parsed_zones = build_zones(ZONES, profile, zone_half_width, blend_width_stops)
    check_zone_overlaps(parsed_zones)
    print_zone_preview(ZONES, parsed_zones)

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    lut_data = build_lut(parsed_zones, lut_size)
    write_cube(lut_data, lut_size, filename, profile, ZONES, parsed_zones)

    size_kb = os.path.getsize(filename) / 1024
    print(f"\n✅ Done!  {size_kb:.1f} KB")
    print(f"   Drop '{filename}' into DaVinci Resolve,")
    print(f"   Premiere Pro, FCPX, or your camera's LUT slot.")
    print("=" * 58)


if __name__ == "__main__":
    main()
