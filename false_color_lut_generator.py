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
"""

import math
import os


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
ZONE_HALF_WIDTH_STOPS = 0.30   # Width of each color band (±stops from center)
                                # 0.20 = tight bands,  0.35 = wide bands

BLEND_WIDTH_STOPS = 0.12       # Softness at zone edges
                                # 0.05 = hard edge,  0.20 = very soft

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
OUTPUT_FILENAME = "FalseColor_{profile}.cube"

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

def _vlog_to_log(x):
    """Panasonic V-Log L. Covers ~14 stops dynamic range."""
    x = max(x, 1e-10)
    if x < 0.01:
        return 5.6 * x + 0.125
    return 0.241514 * math.log10(x + 0.00873) + 0.598206


def _slog3_to_log(x):
    """Sony S-Log3. Covers ~15.3 stops dynamic range."""
    x = max(x, 1e-10)
    if x >= 0.014:
        return (420.0 + math.log10((x + 0.01) / 0.20) * 261.5) / 1023.0
    return (x * (171.2102946929 - 95.0) / 0.01 + 95.0) / 1023.0


def _logc3_to_log(x):
    """ARRI LogC3 at EI800. Covers ~14 stops dynamic range."""
    x = max(x, 1e-10)
    if x >= 0.010591:
        return 0.247190 * math.log10(5.555556 * x + 0.052272) + 0.385537
    return 5.367655 * x + 0.092809


def _clog2_to_log(x):
    """Canon C-Log2. Covers ~15+ stops dynamic range."""
    x = max(x, 1e-10)
    return 0.281863093 * math.log10(87.09937546 * x + 1.0) + 0.073059361


def _flog2_to_log(x):
    """
    Fuji F-Log2. Covers ~14+ stops dynamic range.
    Reference: Fujifilm F-Log2 Specification v1.0
    """
    x = max(x, 1e-10)
    cut = 0.000889544
    if x >= cut:
        return 0.384736 * math.log10(x + 0.064829) + 0.553667
    return 8.799461 * x + 0.092864


def _nlog_to_log(x):
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


def _bmpfilm5_to_log(x):
    """
    Blackmagic Film Gen 5. Covers ~13 stops dynamic range.
    Reference: Blackmagic Design Generation 5 Color Science SDK.
    """
    x = max(x, 1e-10)
    cut = 0.005
    if x < cut:
        return 8.283605932402494 * x + 0.09246575342465753
    return 0.08692876065491224 * math.log(x + 0.005494072432257808) + 0.5402385771788551


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
        "black_clip":    (0.00, 0.10),
    },
    "slog3": {
        "name":          "Sony S-Log3",
        "cameras":       "A7S III, A7 IV, FX3, FX6, FX9, ZV-E1, Venice",
        "linear_to_log": _slog3_to_log,
        "middle_gray":   0.4049,
        "white_clip":    (0.76, 1.00),
        "black_clip":    (0.00, 0.09),
    },
    "logc3": {
        "name":          "ARRI LogC3 (EI800)",
        "cameras":       "Alexa Mini, Alexa Mini LF, Alexa 35, Amira",
        "linear_to_log": _logc3_to_log,
        "middle_gray":   0.3910,
        "white_clip":    (0.75, 1.00),
        "black_clip":    (0.00, 0.09),
    },
    "clog2": {
        "name":          "Canon C-Log2",
        "cameras":       "C70, C300 III, C500 II, EOS R5 C, EOS C70",
        "linear_to_log": _clog2_to_log,
        "middle_gray":   0.4175,
        "white_clip":    (0.70, 1.00),
        "black_clip":    (0.00, 0.07),
    },
    "flog2": {
        "name":          "Fuji F-Log2",
        "cameras":       "X-H2S, X-H2, GFX100S II",
        "linear_to_log": _flog2_to_log,
        "middle_gray":   0.3185,
        "white_clip":    (0.60, 1.00),
        "black_clip":    (0.00, 0.09),
    },
    "nlog": {
        "name":          "Nikon N-Log  [approximate]",
        "cameras":       "Z6, Z7, Z6II, Z7II, Z8, Z9",
        "linear_to_log": _nlog_to_log,
        "middle_gray":   0.3635,
        "white_clip":    (0.72, 1.00),
        "black_clip":    (0.00, 0.08),
    },
    "bmpfilm5": {
        "name":          "Blackmagic Film Gen 5",
        "cameras":       "Pocket 6K G2, 6K Pro, 6K G2, URSA Mini Pro 12K",
        "linear_to_log": _bmpfilm5_to_log,
        "middle_gray":   0.3938,
        "white_clip":    (0.75, 1.00),
        "black_clip":    (0.00, 0.09),
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

def stop_to_log(stops, profile):
    """
    Convert a stop offset (relative to middle gray) to a log-encoded value.
    Each stop doubles or halves the linear light level:
      +1 stop = 0.18 * 2^1 = 0.36 linear
      -2 stops = 0.18 * 2^-2 = 0.045 linear
    """
    linear = MIDDLE_GRAY_LINEAR * (2.0 ** stops)
    return profile["linear_to_log"](linear)


def stop_to_log_range(stops, half_width_stops, profile):
    """
    Return (lo, hi) log boundaries for a zone centered at 'stops',
    spanning ±half_width_stops on either side.
    """
    lo = stop_to_log(stops - half_width_stops, profile)
    hi = stop_to_log(stops + half_width_stops, profile)
    return lo, hi


def blend_width_in_log(blend_width_stops, profile):
    """
    Convert a stop-based blend width to log units by measuring
    how wide one stop is in log space at middle gray.
    """
    center       = stop_to_log(0, profile)
    one_stop_up  = stop_to_log(blend_width_stops, profile)
    return abs(one_stop_up - center)


# ───────────────────────────────────────────────────────────────────────
#  ZONE BUILDER
# ───────────────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color):
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


def build_zones(zones_config, profile, half_width_stops):
    """
    Resolve each entry in the user's ZONES list into a
    (lo, hi, r, g, b) tuple with log-domain boundaries.
    """
    parsed = []
    for stop_val, hex_color in zones_config:
        r, g, b = hex_to_rgb(hex_color)

        if stop_val == "white_clip":
            lo, hi = profile["white_clip"]
        elif stop_val == "black_clip":
            lo, hi = profile["black_clip"]
        elif isinstance(stop_val, (int, float)):
            lo, hi = stop_to_log_range(stop_val, half_width_stops, profile)
        else:
            raise ValueError(
                f"Unknown zone value '{stop_val}' — "
                f"use a number, 'white_clip', or 'black_clip'"
            )

        parsed.append((lo, hi, r, g, b))

    return parsed


# ───────────────────────────────────────────────────────────────────────
#  CORE FALSE COLOR MATH
# ───────────────────────────────────────────────────────────────────────

def rgb_luminance(r, g, b):
    """
    Rec.709 perceptual luma. Combines R, G, B into a single brightness.
    Green is weighted most because human eyes are most sensitive to it.
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def smoothstep(edge0, edge1, x):
    """S-curve interpolation: 0.0 at edge0, 1.0 at edge1, smooth in between."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def apply_false_color(r, g, b, parsed_zones, blend_width):
    """
    Given log-encoded R,G,B (0.0–1.0):
    - Calculate luma
    - Find the zone with the strongest match (with smooth edges)
    - Return the blended false color, or grayscale if outside all zones
    """
    luma        = rgb_luminance(r, g, b)
    best_weight = 0.0
    best_color  = None

    for (lo, hi, fr, fg, fb) in parsed_zones:
        fade_in  = smoothstep(lo - blend_width, lo + blend_width, luma)
        fade_out = 1.0 - smoothstep(hi - blend_width, hi + blend_width, luma)
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

def build_lut(parsed_zones, size, blend_width):
    """Build the full size³ LUT table."""
    lut  = []
    step = 1.0 / (size - 1)
    print(f"  Building {size}³ LUT ({size**3:,} entries)...")

    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r_out, g_out, b_out = apply_false_color(
                    ri * step, gi * step, bi * step,
                    parsed_zones, blend_width
                )
                lut.append((
                    max(0.0, min(1.0, r_out)),
                    max(0.0, min(1.0, g_out)),
                    max(0.0, min(1.0, b_out)),
                ))

    print(f"  ✓ {len(lut):,} entries generated.")
    return lut


def write_cube(lut, size, filename, profile, zones_config, parsed_zones):
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
        f"# Generator: false_color_generator.py\n"
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

def print_zone_preview(zones_config, parsed_zones, blend_width):
    print(f"\n  {'Zone':<16} {'Log Range':<18} {'Target':<10} {'Output':<10} Match")
    print(f"  {'────':<16} {'─────────':<18} {'──────':<10} {'──────':<10} ─────")

    for (stop_val, hex_col), (lo, hi, fr, fg, fb) in zip(zones_config, parsed_zones):
        label = (
            "white_clip" if stop_val == "white_clip" else
            "black_clip" if stop_val == "black_clip" else
            f"{stop_val:+.0f} stops"
        )
        center = (lo + hi) / 2.0
        r_out, g_out, b_out = apply_false_color(
            center, center, center, parsed_zones, blend_width
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        match = "✓" if hex_out.upper() == hex_col.upper() else "~"
        print(f"  {label:<16} {lo:.3f} - {hi:.3f}    {hex_col:<10} {hex_out:<10} {match}")

    # Sample a between-zone point (should always be gray)
    if len(parsed_zones) >= 2:
        gap_mid = (parsed_zones[0][1] + parsed_zones[1][0]) / 2.0
        r_out, g_out, b_out = apply_false_color(
            gap_mid, gap_mid, gap_mid, parsed_zones, blend_width
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        print(f"  {'(between zones)':<16} {gap_mid:.3f}              {'(gray)':<10} {hex_out:<10} –")

    print()


# ───────────────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Validate profile
    if LOG_PROFILE not in PROFILES:
        available = "  |  ".join(PROFILES.keys())
        raise ValueError(
            f"\nUnknown profile '{LOG_PROFILE}'.\n"
            f"Available options:  {available}\n"
        )

    profile    = PROFILES[LOG_PROFILE]
    filename   = OUTPUT_FILENAME.replace("{profile}", LOG_PROFILE)
    blend_log  = blend_width_in_log(BLEND_WIDTH_STOPS, profile)

    print("=" * 58)
    print("  False Color Exposure LUT Generator")
    print("=" * 58)
    print(f"\n  Profile       : {profile['name']}")
    print(f"  Cameras       : {profile['cameras']}")
    print(f"  Middle gray   : {profile['middle_gray']:.4f}  (log-encoded 18% gray)")
    print(f"  Zones defined : {len(ZONES)}")
    print(f"  Zone width    : ±{ZONE_HALF_WIDTH_STOPS} stops")
    print(f"  Blend width   : {BLEND_WIDTH_STOPS} stops  ({blend_log:.4f} log units)")
    print(f"  LUT size      : {LUT_SIZE}³  ({LUT_SIZE**3:,} entries)")
    print(f"  Output file   : {filename}")

    parsed_zones = build_zones(ZONES, profile, ZONE_HALF_WIDTH_STOPS)
    print_zone_preview(ZONES, parsed_zones, blend_log)

    lut_data = build_lut(parsed_zones, LUT_SIZE, blend_log)
    write_cube(lut_data, LUT_SIZE, filename, profile, ZONES, parsed_zones)

    size_kb = os.path.getsize(filename) / 1024
    print(f"\n✅ Done!  {size_kb:.1f} KB")
    print(f"   Drop '{filename}' into DaVinci Resolve,")
    print(f"   Premiere Pro, FCPX, or your camera's LUT slot.")
    print("=" * 58)