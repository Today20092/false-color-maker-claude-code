"""
Panasonic V-Log False Color Exposure LUT Generator
====================================================
Edit the CONFIG section below to customize your zones.
Everything else is automatic.
"""

# ═══════════════════════════════════════════════════════════════
#  CONFIG — Edit this section only
# ═══════════════════════════════════════════════════════════════

# LUT grid size. 33 is standard. 65 is higher quality but larger file.
LUT_SIZE = 33

# How wide the blend/feather is at each zone edge (in V-Log units).
# 0.005 = very tight, 0.015 = soft, 0.030 = very wide/dreamy
BLEND_WIDTH = 0.009

# Output filename
OUTPUT_FILENAME = "VLog_FalseColor_Exposure3.cube"

# Zone definitions — add, remove, or edit rows freely.
# Format per row:
#   ( lo,   hi,   "HEX COLOR" )
#    ↑       ↑         ↑
#  V-Log  V-Log    HTML hex color code
#  min    max      (with or without the # symbol)
#
# Anything outside all zones becomes grayscale.

ZONES = [
    (0.80, 1.01, "#ef4444"),   # White Clip
    (0.56, 0.60, "#fde047"),   # +2 Stops
    (0.48, 0.52, "#d946ef"),   # +1 Stop
    (0.40, 0.44, "#22c55e"),   # Mid Gray
    (0.32, 0.36, "#06b6d4"),   # -1 Stop
    (0.24, 0.28, "#3b82f6"),   # -2 Stops
    (0.00, 0.10, "#6d28d9"),   # Black Clip
]

# ═══════════════════════════════════════════════════════════════
#  END OF CONFIG — No need to edit below this line
# ═══════════════════════════════════════════════════════════════

import numpy as np


# ── Hex → normalized RGB (0.0–1.0) ──────────────────────────────
def hex_to_rgb(hex_color):
    """Convert a hex color string like '#FF0000' or 'FF0000' to (r, g, b) floats."""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: '#{hex_color}' — must be 6 digits e.g. #FF0000")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


# ── Build parsed zones from config ──────────────────────────────
def build_parsed_zones(zones):
    """
    Convert the human-friendly ZONES list into a list of
    (lo, hi, r, g, b) tuples ready for the LUT math.
    """
    parsed = []
    for i, entry in enumerate(zones):
        if len(entry) != 3:
            raise ValueError(f"Zone {i+1} must have exactly 3 values: (lo, hi, hex_color)")
        lo, hi, hex_color = entry
        r, g, b = hex_to_rgb(hex_color)
        parsed.append((lo, hi, r, g, b))
    return parsed


# ── V-Log luma extraction ────────────────────────────────────────
def rgb_luminance(r, g, b):
    """
    Combine R, G, B into a single perceptual brightness value.
    Uses Rec.709 luma weights — green contributes most because
    human eyes are most sensitive to green light.
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ── Smoothstep blend curve ───────────────────────────────────────
def smoothstep(edge0, edge1, x):
    """
    Returns a smooth 0→1 value as x moves from edge0 to edge1.
    Uses an S-curve so transitions ease in and out naturally.
    """
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


# ── Core false color logic ───────────────────────────────────────
def apply_false_color(r, g, b, parsed_zones, blend_width):
    """
    Given V-Log R,G,B values (0–1):
    - Calculate luma
    - Find which zone it belongs to (with soft edges)
    - Return blended false color, or grayscale if outside all zones
    """
    luma = rgb_luminance(r, g, b)

    best_color = None
    best_weight = 0.0

    for (lo, hi, fr, fg, fb) in parsed_zones:
        fade_in  = smoothstep(lo - blend_width, lo + blend_width, luma)
        fade_out = 1.0 - smoothstep(hi - blend_width, hi + blend_width, luma)
        weight   = fade_in * fade_out

        if weight > best_weight:
            best_weight = weight
            best_color  = (fr, fg, fb)

    if best_color is None or best_weight == 0.0:
        return luma, luma, luma   # grayscale passthrough

    fr, fg, fb = best_color
    r_out = fr * best_weight + luma * (1.0 - best_weight)
    g_out = fg * best_weight + luma * (1.0 - best_weight)
    b_out = fb * best_weight + luma * (1.0 - best_weight)
    return r_out, g_out, b_out


# ── Build the full 3D LUT table ──────────────────────────────────
def build_lut(parsed_zones, size, blend_width):
    lut   = []
    step  = 1.0 / (size - 1)

    print(f"Building {size}³ LUT ({size**3:,} entries)...")

    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r_in = ri * step
                g_in = gi * step
                b_in = bi * step

                r_out, g_out, b_out = apply_false_color(
                    r_in, g_in, b_in, parsed_zones, blend_width
                )

                lut.append((
                    max(0.0, min(1.0, r_out)),
                    max(0.0, min(1.0, g_out)),
                    max(0.0, min(1.0, b_out)),
                ))

    print(f"  ✓ {len(lut):,} entries generated.")
    return lut


# ── Write .cube file ─────────────────────────────────────────────
def write_cube(lut, size, filename, zones):
    zone_comments = "\n".join(
        f"#   {hex_col:<10}  {lo:.2f} - {hi:.2f}"
        for (lo, hi, hex_col) in zones
    )

    header = f"""# Panasonic V-Log False Color Exposure Monitor LUT
# Generated by vlog_false_color_generator.py
#
# Zones:
{zone_comments}
#   Grayscale   = between zones
#
TITLE "VLog FalseColor Exposure Monitor"
LUT_3D_SIZE {size}
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

"""
    with open(filename, "w") as f:
        f.write(header)
        for (r, g, b) in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    print(f"  ✓ Written to: {filename}")


# ── Zone preview printout ────────────────────────────────────────
def print_zone_preview(parsed_zones, blend_width):
    print("\n-- Zone Preview -----------------------------------------------")
    print(f"  {'V-Log':>6}  {'Input Label':<30}  Output Hex   RGB")
    print(f"  {'------':>6}  {'------------------------------':<30}  ----------  ------------------")

    test_points = []
    for (lo, hi, fr, fg, fb) in parsed_zones:
        center = (lo + hi) / 2.0
        hex_in = "#{:02X}{:02X}{:02X}".format(
            int(fr * 255), int(fg * 255), int(fb * 255)
        )
        test_points.append((center, f"Zone center  (target {hex_in})"))

    for i in range(len(parsed_zones) - 1):
        hi_current = parsed_zones[i][1]
        lo_next    = parsed_zones[i + 1][0]
        mid        = (hi_current + lo_next) / 2.0
        test_points.append((mid, "Between zones (expect gray)"))

    test_points.sort(key=lambda x: x[0], reverse=True)

    for val, label in test_points:
        r_out, g_out, b_out = apply_false_color(val, val, val, parsed_zones, blend_width)
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        print(f"  {val:>6.3f}  {label:<30}  {hex_out}    ({r_out:.2f}, {g_out:.2f}, {b_out:.2f})")
    print()


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("=" * 57)
    print("  V-Log False Color Exposure LUT Generator")
    print("=" * 57)
    print(f"\n  Zones defined : {len(ZONES)}")
    print(f"  LUT size      : {LUT_SIZE}^3 ({LUT_SIZE**3:,} entries)")
    print(f"  Blend width   : {BLEND_WIDTH}")
    print(f"  Output file   : {OUTPUT_FILENAME}")

    parsed_zones = build_parsed_zones(ZONES)

    print_zone_preview(parsed_zones, BLEND_WIDTH)

    lut_data = build_lut(parsed_zones, LUT_SIZE, BLEND_WIDTH)
    write_cube(lut_data, LUT_SIZE, OUTPUT_FILENAME, ZONES)

    size_kb = os.path.getsize(OUTPUT_FILENAME) / 1024
    print(f"\n Done!  File size: {size_kb:.1f} KB")
    print(f"   Drop '{OUTPUT_FILENAME}' into DaVinci Resolve,")
    print(f"   Premiere Pro, or your Lumix camera's LUT slot.")
    print("=" * 57)