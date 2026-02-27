"""
Hybrid Monitor LUT Generator — Sony S-Log3 / S-Gamut3.Cine
===========================================================
Generates a 33x33x33 .cube LUT that provides:
  - Standard S-Log3 → Rec.709 base transform (full color, not grayscale)
  - Narrow "exposure tracer" overlays at key IRE zones
  - ~90% of the image remains in Rec.709 for white balance judging

Overlay zones (keyed to S-Log3 signal level in IRE):
   0 –   2 IRE  → Solid Blue   (0, 0, 1)   — clipping blacks
  40 –  42 IRE  → Solid Green  (0, 1, 0)   — middle gray 18%  (~41 IRE in S-Log3)
  52 –  56 IRE  → Solid Pink   (1, .4, .7) — skin tones +1 stop
  94 – 100 IRE  → Solid Red    (1, 0, 0)   — highlight clipping

Run with:
  uv run hybrid_monitor_lut.py
"""

import math
import os

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

LUT_SIZE        = 33
OUTPUT_FILENAME = "luts/HybridMonitor_SLog3_SGamut3Cine_Rec709.cube"

# Overlay zones: (ire_lo, ire_hi, r, g, b)
# IRE = normalized S-Log3 code value × 100
# S-Log3 reference points:
#   Optical black  ≈  9 IRE  (95/1023)
#   Middle gray    ≈ 41 IRE  (420/1023)  ← zone center at 41
#   White clip     ≈ 94 IRE
HYBRID_ZONES = [
    (  0.0,   2.0,  0.0, 0.0, 1.0),   # Black clip  → Blue
    ( 40.0,  42.0,  0.0, 1.0, 0.0),   # Middle gray → Green
    ( 52.0,  56.0,  1.0, 0.4, 0.7),   # Skin +1stop → Pink
    ( 94.0, 101.0,  1.0, 0.0, 0.0),   # Hi clip     → Red   (101 = open upper bound)
]

# Smoothstep feather at each zone edge, in IRE units.
# 0.5 = gentle half-IRE ramp (recommended for clean but non-clinical edges)
# 0.0 = binary hard cut
EDGE_FEATHER_IRE = 0.5


# ═══════════════════════════════════════════════════════════════════════
#  S-LOG3 TRANSFER FUNCTIONS  (Sony S-Log3 Technical Summary v1.0)
# ═══════════════════════════════════════════════════════════════════════

def _slog3_linear_to_log(x):
    """Scene-linear (0.18 = 18% gray) → S-Log3 encoded (0–1)."""
    x = max(x, 1e-10)
    if x >= 0.01125:
        return (420.0 + math.log10((x + 0.01) / 0.19) * 261.5) / 1023.0
    return (x * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0


# Cutpoint in log space where the two S-Log3 segments meet
_SLOG3_CUT_LOG = (171.2102946929) / 1023.0  # ≈ 0.16737


def _slog3_log_to_linear(v):
    """S-Log3 encoded (0–1) → scene-linear. Exact inverse of the above."""
    if v >= _SLOG3_CUT_LOG:
        return 0.19 * (10.0 ** ((v * 1023.0 - 420.0) / 261.5)) - 0.01
    return (v * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)


# ═══════════════════════════════════════════════════════════════════════
#  S-GAMUT3.CINE → REC.709 COLOR MATRIX  (linear scene light)
# ═══════════════════════════════════════════════════════════════════════
# Derived from:
#   S-Gamut3.Cine primaries  (Sony S-Gamut3.Cine specification, D65)
#   Rec.709 primaries        (ITU-R BT.709, D65)
# Consistent with DaVinci Resolve "S-Gamut3.Cine / S-Log3" input transform.

_M = [
    [ 1.8467890, -0.7883606, -0.0584283],
    [-0.3849882,  1.3640906,  0.0208976],
    [ 0.0194736, -0.1233739,  1.1039003],
]


def _sgamut3cine_to_rec709_linear(r, g, b):
    """Apply the 3×3 S-Gamut3.Cine → Rec.709 matrix in linear light."""
    ro = _M[0][0] * r + _M[0][1] * g + _M[0][2] * b
    go = _M[1][0] * r + _M[1][1] * g + _M[1][2] * b
    bo = _M[2][0] * r + _M[2][1] * g + _M[2][2] * b
    return ro, go, bo


# ═══════════════════════════════════════════════════════════════════════
#  REC.709 OETF  (linear light → gamma-encoded display signal)
#  ITU-R BT.709, Section 2
# ═══════════════════════════════════════════════════════════════════════

def _linear_to_rec709(x):
    x = max(x, 0.0)
    if x < 0.018:
        return 4.5 * x
    return 1.099 * (x ** 0.45) - 0.099


# ═══════════════════════════════════════════════════════════════════════
#  FULL BASE TRANSFORM: S-Log3/S-Gamut3.Cine → Rec.709
# ═══════════════════════════════════════════════════════════════════════

def slog3_sgamut3cine_to_rec709(r_in, g_in, b_in):
    """
    Three-stage pipeline for the S-Log3 → Rec.709 base transform:
      1. S-Log3 log-decode  → scene-linear per channel
      2. S-Gamut3.Cine → Rec.709 matrix  (linear light)
      3. Rec.709 OETF gamma encode
    """
    # Stage 1 — log to linear
    r_lin = _slog3_log_to_linear(r_in)
    g_lin = _slog3_log_to_linear(g_in)
    b_lin = _slog3_log_to_linear(b_in)

    # Stage 2 — gamut matrix
    r_709, g_709, b_709 = _sgamut3cine_to_rec709_linear(r_lin, g_lin, b_lin)

    # Stage 3 — gamma encode
    return (
        max(0.0, min(1.0, _linear_to_rec709(r_709))),
        max(0.0, min(1.0, _linear_to_rec709(g_709))),
        max(0.0, min(1.0, _linear_to_rec709(b_709))),
    )


# ═══════════════════════════════════════════════════════════════════════
#  LUMINANCE PROBE — S-Log3 IRE level of a pixel
# ═══════════════════════════════════════════════════════════════════════

def _slog3_ire(r_in, g_in, b_in):
    """
    Compute the S-Log3 luminance in IRE (0–100) for zone detection.

    Uses the weighted log-channel average (Rec.709 luma coefficients applied
    directly to the log-encoded values). This matches the code-value percentage
    displayed on a waveform monitor and correctly represents sub-black values
    (< 9 IRE) without clamping — which is essential for the black-clip zone.

    For neutral/gray pixels (R≈G≈B) this is exact. For saturated colors the
    luma coefficients provide the same perceptual weighting used by Rec.709
    display systems, which is appropriate for exposure monitoring.
    """
    return (0.2126 * r_in + 0.7152 * g_in + 0.0722 * b_in) * 100.0


# ═══════════════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _smoothstep(edge0, edge1, x):
    """S-curve interpolation: 0.0 at edge0, 1.0 at edge1."""
    if edge0 >= edge1:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


# ═══════════════════════════════════════════════════════════════════════
#  HYBRID OVERLAY LOGIC
# ═══════════════════════════════════════════════════════════════════════

def apply_hybrid(r_in, g_in, b_in, zones, feather_ire):
    """
    For each LUT coordinate (r_in, g_in, b_in) in S-Log3 / S-Gamut3.Cine:

      1. Compute the Rec.709 base color (full color transform, not grayscale).
      2. Compute the pixel's S-Log3 luminance in IRE.
      3. If the IRE falls inside an overlay zone, blend toward the zone color.
         The blend weight uses smoothstep ramps at the zone edges so the
         transition is clean but not clinically hard.
      4. Outside all zones → return pure Rec.709 color unmodified.
    """
    # Step 1 — Rec.709 base (this is what 90%+ of pixels will output)
    r_base, g_base, b_base = slog3_sgamut3cine_to_rec709(r_in, g_in, b_in)

    # Step 2 — luminance probe (IRE 0–100)
    ire = _slog3_ire(r_in, g_in, b_in)

    # Step 3 — find the strongest overlapping zone
    best_weight = 0.0
    best_rgb    = None

    for (lo, hi, fr, fg, fb) in zones:
        fade_in  = _smoothstep(lo - feather_ire, lo + feather_ire, ire)
        fade_out = 1.0 - _smoothstep(hi - feather_ire, hi + feather_ire, ire)
        weight   = fade_in * fade_out

        if weight > best_weight:
            best_weight = weight
            best_rgb    = (fr, fg, fb)

    # Step 4 — no zone matched → pure Rec.709 passthrough
    if best_rgb is None or best_weight == 0.0:
        return r_base, g_base, b_base

    fr, fg, fb = best_rgb
    return (
        fr * best_weight + r_base * (1.0 - best_weight),
        fg * best_weight + g_base * (1.0 - best_weight),
        fb * best_weight + b_base * (1.0 - best_weight),
    )


# ═══════════════════════════════════════════════════════════════════════
#  LUT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_hybrid_lut(size, zones, feather_ire):
    """Build the full size³ LUT. Scan order: R fast, B slow (standard .cube)."""
    lut  = []
    step = 1.0 / (size - 1)
    print(f"  Building {size}³ LUT ({size**3:,} entries)...")

    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r_out, g_out, b_out = apply_hybrid(
                    ri * step, gi * step, bi * step,
                    zones, feather_ire,
                )
                lut.append((
                    max(0.0, min(1.0, r_out)),
                    max(0.0, min(1.0, g_out)),
                    max(0.0, min(1.0, b_out)),
                ))

    print(f"  ✓ {len(lut):,} entries generated.")
    return lut


# ═══════════════════════════════════════════════════════════════════════
#  CUBE FILE WRITER
# ═══════════════════════════════════════════════════════════════════════

def write_hybrid_cube(lut, size, filename, zones):
    zone_lines = "\n".join(
        "#   {:5.1f} – {:5.1f} IRE  →  RGB({:.1f}, {:.2f}, {:.1f})  #{:02X}{:02X}{:02X}".format(
            lo, hi, fr, fg, fb,
            int(fr * 255), int(fg * 255), int(fb * 255),
        )
        for (lo, hi, fr, fg, fb) in zones
    )

    header = (
        "# Hybrid Monitor LUT — Sony A7S III\n"
        "# Input  : S-Log3 / S-Gamut3.Cine\n"
        "# Output : Rec.709 base + narrow exposure tracer overlays\n"
        "# Generator: hybrid_monitor_lut.py\n"
        "#\n"
        "# Base transform pipeline:\n"
        "#   S-Log3 log-decode  →  S-Gamut3.Cine→Rec.709 matrix  →  Rec.709 OETF\n"
        "#\n"
        "# Overlay zones (S-Log3 IRE):\n"
        f"{zone_lines}\n"
        "#   All other IRE values → standard Rec.709 color (no tracer)\n"
        "#\n"
        f"TITLE \"HybridMonitor SLog3 SGamut3Cine Rec709\"\n"
        f"LUT_3D_SIZE {size}\n"
        "DOMAIN_MIN 0.0 0.0 0.0\n"
        "DOMAIN_MAX 1.0 1.0 1.0\n\n"
    )

    with open(filename, "w") as f:
        f.write(header)
        for (r, g, b) in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    print(f"  ✓ Written to: {filename}")


# ═══════════════════════════════════════════════════════════════════════
#  ZONE PREVIEW (console)
# ═══════════════════════════════════════════════════════════════════════

_ZONE_LABELS = {
    ( 0.0,  2.0): "Black clip",
    (40.0, 42.0): "Middle gray",
    (52.0, 56.0): "Skin tone +1",
    (94.0,101.0): "Hi clip",
}

def print_hybrid_preview(zones, feather_ire):
    print(f"\n  {'Zone':<16} {'IRE Range':<14} {'Target':<10} {'Output @ mid':<12} Match")
    print(f"  {'────':<16} {'─────────':<14} {'──────':<10} {'────────────':<12} ─────")

    for (lo, hi, fr, fg, fb) in zones:
        target_hex = "#{:02X}{:02X}{:02X}".format(int(fr*255), int(fg*255), int(fb*255))
        hi_probe   = min(hi, 100.0)           # clamp 101 open-bound for preview
        mid_ire    = (lo + hi_probe) / 2.0
        mid_log    = mid_ire / 100.0
        r_out, g_out, b_out = apply_hybrid(mid_log, mid_log, mid_log, zones, feather_ire)
        out_hex    = "#{:02X}{:02X}{:02X}".format(int(r_out*255), int(g_out*255), int(b_out*255))
        match      = "✓" if target_hex.upper() == out_hex.upper() else "~"
        label      = _ZONE_LABELS.get((lo, hi), f"{lo:.0f}–{hi:.0f} IRE")
        print(f"  {label:<16} {lo:5.1f} – {hi_probe:5.1f}    {target_hex:<10} {out_hex:<12} {match}")

    # Sample a between-zone point — should be pure Rec.709, no tracer
    for sample_ire, sample_label in [(5.0, "5 IRE (near blk)"),
                                     (20.0, "20 IRE (shadows)"),
                                     (70.0, "70 IRE (highlights)")]:
        s = sample_ire / 100.0
        r_out, g_out, b_out = apply_hybrid(s, s, s, zones, feather_ire)
        out_hex = "#{:02X}{:02X}{:02X}".format(int(r_out*255), int(g_out*255), int(b_out*255))
        print(f"  {sample_label:<16} {sample_ire:5.1f} IRE           {'(Rec.709)':<10} {out_hex:<12} –")

    print()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    total_zone_ire = sum(min(hi, 100.0) - lo for lo, hi, *_ in HYBRID_ZONES)
    usable_ire     = 100.0 - 9.0   # S-Log3 legal range: black ~9 IRE to 100 IRE
    pct_colored    = total_zone_ire / 100.0 * 100.0
    pct_rec709     = 100.0 - pct_colored

    print("=" * 62)
    print("  Hybrid Monitor LUT Generator")
    print("  Sony S-Log3 / S-Gamut3.Cine  →  Rec.709 + Tracers")
    print("=" * 62)
    print(f"\n  Camera        : Sony A7S III (S-Log3 / S-Gamut3.Cine)")
    print(f"  Base output   : S-Log3 → Rec.709 (full color, not grayscale)")
    print(f"  LUT size      : {LUT_SIZE}³  ({LUT_SIZE**3:,} entries)")
    print(f"  Output file   : {OUTPUT_FILENAME}")
    print(f"\n  Overlay zones:")

    for lo, hi, fr, fg, fb in HYBRID_ZONES:
        hex_col = "#{:02X}{:02X}{:02X}".format(int(fr*255), int(fg*255), int(fb*255))
        label   = _ZONE_LABELS.get((lo, hi), f"custom")
        print(f"    {lo:5.1f} – {min(hi,100.0):5.1f} IRE  →  {hex_col}  ({label})")

    print(f"\n  Edge feathering : ±{EDGE_FEATHER_IRE} IRE (smoothstep)")
    print(f"  Signal colored  : {total_zone_ire:.1f} / 100 IRE = {pct_colored:.0f}%")
    print(f"  Signal Rec.709  : {pct_rec709:.0f}%  {'✓ meets target' if pct_rec709 >= 85 else '⚠ below 85% target'}")
    print(f"\n  S-Log3 reference:")
    print(f"    Optical black : ~9 IRE  (95/1023)")
    print(f"    Middle gray   : ~41 IRE (420/1023)  ← green zone center")
    print(f"    White clip    : ~94 IRE")

    print_hybrid_preview(HYBRID_ZONES, EDGE_FEATHER_IRE)

    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    lut_data = build_hybrid_lut(LUT_SIZE, HYBRID_ZONES, EDGE_FEATHER_IRE)
    write_hybrid_cube(lut_data, LUT_SIZE, OUTPUT_FILENAME, HYBRID_ZONES)

    size_kb = os.path.getsize(OUTPUT_FILENAME) / 1024
    print(f"\n✅ Done!  {size_kb:.1f} KB")
    print(f"   Load '{OUTPUT_FILENAME}' into:")
    print(f"   • Camera LUT slot (A7S III: Menu → Shooting → LUT)")
    print(f"   • DaVinci Resolve node (input transform: S-Log3/S-Gamut3.Cine)")
    print(f"   • SmallHD / Atomos monitor LUT slot")
    print("=" * 62)
