"""
Hybrid Monitor LUT Generator
==============================
Generates a .cube LUT that combines two things in a single pass:

  1. Full camera log → Rec.709 base transform  (preserves color for white balance)
  2. Thin false color "tracer" slices at key exposure stop zones

The false color bands are intentionally narrow (default ±0.07 stops) so that
~90%+ of the image stays in clean Rec.709. You can judge white balance and
color while the tracers flag overexposure, middle gray, and crushed blacks —
without turning off the overlay or switching LUT slots.

Supported profiles (same as false_color_lut_generator.py):
  vlog | slog3 | logc3 | clog2 | flog2 | nlog | bmpfilm5

Run with:
  uv run hybrid_monitor_lut.py
  uv run hybrid_monitor_lut.py --profile vlog --size 65
  uv run hybrid_monitor_lut.py --profile slog3 --half-width 0.05

Dependencies: colour-science, numpy, pillow, typer, rich
"""

import os
from typing import Annotated, Optional

import colour
import numpy as np
import typer
from PIL import Image
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from false_color_lut_generator import (
    PROFILES,
    build_zones,
    check_zone_overlaps,
    smoothstep,
    _swatch,
    console,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG — Only edit this section
# ═══════════════════════════════════════════════════════════════════════

# ── Default camera log profile ─────────────────────────────────────────
# Options:  vlog | slog3 | logc3 | clog2 | flog2 | nlog | bmpfilm5
LOG_PROFILE = "slog3"

# ── LUT settings ───────────────────────────────────────────────────────
LUT_SIZE = 33

# ── Hybrid zone width (MUCH narrower than pure false color) ────────────
#
# The whole point of a hybrid LUT is that most pixels render in Rec.709.
# Keep ZONE_HALF_WIDTH_STOPS small so the colored bands are thin tracers.
#
#   half_width   zone width   approx % of signal colored (7 zones, ~10 stops)
#   0.04         0.08 stops   ~6%   ← very thin, clinical
#   0.07         0.14 stops   ~10%  ← default sweet spot
#   0.10         0.20 stops   ~15%  ← wider; easier to see on small monitors
#   0.15         0.30 stops   ~21%  ← same as pure false color; loses Rec.709 feel
#
ZONE_HALF_WIDTH_STOPS = 0.07   # Default: thin tracer bands

BLEND_WIDTH_STOPS = 0.02       # Feathering at zone edges (0.02/0.07 = 29% — good)

# ── Zone definitions ───────────────────────────────────────────────────
# Same stop positions as false_color_lut_generator.py.
# Colors are the same so muscle memory transfers between LUT modes.

ZONES = [
    ("white_clip",  "#ef4444"),   # Clipping highlights → Red
    ( 2,            "#eab308"),   # +2 Stops            → Yellow
    ( 1,            "#d946ef"),   # +1 Stop             → Magenta
    ( 0,            "#22c55e"),   # Mid Gray (0 stops)  → Green
    (-1,            "#06b6d4"),   # -1 Stop             → Cyan
    (-2,            "#3b82f6"),   # -2 Stops            → Blue
    ("black_clip",  "#6366f1"),   # Crushed blacks      → Indigo
]

# ── Output filename ────────────────────────────────────────────────────
OUTPUT_FILENAME = "luts/HybridMonitor_{profile}.cube"

# ═══════════════════════════════════════════════════════════════════════
#  END OF CONFIG — No need to edit below this line
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  CAMERA GAMUT → REC.709 MATRICES
#
#  Each camera records in its own wide-gamut color space.  To produce a
#  faithful Rec.709 base image we must rotate the primaries in linear
#  light before applying the Rec.709 OETF (gamma).
#
#  Matrices are derived from colour-science's colourspace database at
#  module-load time:
#    M = (XYZ→Rec.709) × (CameraGamut→XYZ)
#  Both the camera gamuts and Rec.709 reference D65, so no chromatic
#  adaptation is required.
#
#  If colour-science doesn't know a gamut name, a KeyError is raised with
#  the offending name — run `colour.RGB_COLOURSPACES.keys()` to see all
#  registered colourspaces.
# ───────────────────────────────────────────────────────────────────────

# Map profile key → colour-science colourspace name for the camera gamut
_GAMUT_CS_NAME: dict[str, str] = {
    "vlog":     "V-Gamut",
    "slog3":    "S-Gamut3.Cine",
    "logc3":    "ARRI Wide Gamut 3",
    "clog2":    "Cinema Gamut",
    "flog2":    "F-Gamut",
    "nlog":     "N-Gamut",
    "bmpfilm5": "Blackmagic Wide Gamut",  # colour-science colourspace name
}

_REC709_CS = colour.RGB_COLOURSPACES["ITU-R BT.709"]


def _build_gamut_matrix(gamut_cs_name: str) -> np.ndarray:
    """
    Compute the 3×3 linear matrix: camera gamut → Rec.709.

    Both the camera colourspace and Rec.709 use D65, so the matrix is
    the straight product of (XYZ→Rec.709) × (CameraGamut→XYZ).
    """
    try:
        camera_cs = colour.RGB_COLOURSPACES[gamut_cs_name]
    except KeyError:
        available = ", ".join(list(colour.RGB_COLOURSPACES.keys())[:8]) + "…"
        raise KeyError(
            f"colour-science doesn't have colourspace '{gamut_cs_name}'. "
            f"Some available names: {available}. "
            f"Run `colour.RGB_COLOURSPACES.keys()` for the full list."
        )
    return (_REC709_CS.matrix_XYZ_to_RGB @ camera_cs.matrix_RGB_to_XYZ).astype(np.float64)


# Pre-compute all matrices at import time — fail fast if a name is wrong
_GAMUT_MATRICES: dict[str, np.ndarray] = {
    key: _build_gamut_matrix(name) for key, name in _GAMUT_CS_NAME.items()
}


# ───────────────────────────────────────────────────────────────────────
#  REC.709 OETF
#  Implemented inline so we control the exact formula and it works on
#  both Python floats and NumPy arrays without extra API uncertainty.
# ───────────────────────────────────────────────────────────────────────

def _rec709_oetf_f(x: float) -> float:
    """Rec.709 OETF for a single float (linear → gamma-encoded)."""
    x = max(x, 0.0)
    return 4.5 * x if x < 0.018 else 1.099 * (x ** 0.45) - 0.099


def _rec709_oetf(x: np.ndarray) -> np.ndarray:
    """Rec.709 OETF for a NumPy array (linear → gamma-encoded)."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


# ───────────────────────────────────────────────────────────────────────
#  CORE HYBRID LOGIC — scalar version  (used for zone preview table)
# ───────────────────────────────────────────────────────────────────────

def apply_hybrid_false_color(
    r: float, g: float, b: float,
    parsed_zones: list,
    profile: dict,
    gamut_matrix: np.ndarray,
) -> tuple[float, float, float]:
    """
    Given log-encoded R,G,B (0.0–1.0) in the camera's native gamut:

    Stage A — Base transform (what ~90% of pixels output):
      1. Decode log → scene-linear per channel
      2. Multiply by the camera gamut → Rec.709 matrix (linear light)
      3. Apply Rec.709 OETF (gamma encode for display)

    Stage B — Exposure tracer overlay:
      4. Compute Rec.709 linear luma Y = 0.2126·R + 0.7152·G + 0.0722·B
         on the *linear* decoded signal (physically correct — one stop of
         light is the same fraction of the scale everywhere on the curve)
      5. Re-encode luma back to log for zone threshold comparisons
      6. Find the strongest matching zone (with smooth edges)
      7. Blend zone color over the Rec.709 base by the zone weight

    Outside all zones → return pure Rec.709 color (no false color at all).
    """
    log_to_linear = profile["log_to_linear"]
    linear_to_log = profile["linear_to_log"]

    # ── Stage A: base transform ─────────────────────────────────────
    r_lin = float(log_to_linear(r))
    g_lin = float(log_to_linear(g))
    b_lin = float(log_to_linear(b))

    r_709, g_709, b_709 = gamut_matrix @ np.array([r_lin, g_lin, b_lin])

    r_base = max(0.0, min(1.0, _rec709_oetf_f(float(r_709))))
    g_base = max(0.0, min(1.0, _rec709_oetf_f(float(g_709))))
    b_base = max(0.0, min(1.0, _rec709_oetf_f(float(b_709))))

    # ── Stage B: exposure tracer ────────────────────────────────────
    # Linear luma from the decoded camera signal (correct domain)
    luma_linear = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    luma = float(linear_to_log(max(luma_linear, 1e-10)))

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
        return r_base, g_base, b_base  # ← pure Rec.709 passthrough

    fr, fg, fb = best_color
    return (
        fr * best_weight + r_base * (1.0 - best_weight),
        fg * best_weight + g_base * (1.0 - best_weight),
        fb * best_weight + b_base * (1.0 - best_weight),
    )


# ───────────────────────────────────────────────────────────────────────
#  VECTORISED LUT BUILDER
# ───────────────────────────────────────────────────────────────────────

def build_hybrid_lut(
    parsed_zones: list,
    size: int,
    profile: dict,
    gamut_matrix: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    Build the full size³ LUT using NumPy vectorization.

    Pipeline (vectorised over the entire 3D grid simultaneously):
      1. Decode log R/G/B → linear light  (colour-science, native array support)
      2. Apply 3×3 gamut matrix  (camera → Rec.709, linear light)
      3. Apply Rec.709 OETF  (gamma encode)
      4. Compute linear Rec.709 luma from the decoded linear signal
      5. Re-encode luma to log  (for zone comparisons)
      6. Apply zone smoothstep blending over the Rec.709 base

    Output order: B (outer) → G → R (inner)  — standard .cube scan order.
    """
    console.print(f"  Building [dim]{size}³[/dim] LUT [dim]({size**3:,} entries)[/dim]...")

    vals = np.linspace(0.0, 1.0, size)
    R, G, B = np.meshgrid(vals, vals, vals, indexing="ij")  # [ri, gi, bi]

    log_to_linear = profile["log_to_linear"]
    linear_to_log = profile["linear_to_log"]

    # Step 1 — log → linear (colour-science handles numpy arrays natively)
    R_lin = np.asarray(log_to_linear(R), dtype=np.float64)
    G_lin = np.asarray(log_to_linear(G), dtype=np.float64)
    B_lin = np.asarray(log_to_linear(B), dtype=np.float64)

    # Step 2 — camera gamut → Rec.709 matrix (linear light)
    # Stack into (3, N), multiply, reshape back to (size, size, size)
    shape    = R_lin.shape
    rgb_flat = np.stack([R_lin.ravel(), G_lin.ravel(), B_lin.ravel()])  # (3, N)
    out_flat = gamut_matrix @ rgb_flat                                   # (3, N)
    R_709 = out_flat[0].reshape(shape)
    G_709 = out_flat[1].reshape(shape)
    B_709 = out_flat[2].reshape(shape)

    # Step 3 — Rec.709 OETF (gamma encode for display)
    R_base = np.clip(_rec709_oetf(R_709), 0.0, 1.0)
    G_base = np.clip(_rec709_oetf(G_709), 0.0, 1.0)
    B_base = np.clip(_rec709_oetf(B_709), 0.0, 1.0)

    # Step 4 — linear luma from the original decoded signal (correct domain)
    luma_linear = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin

    # Step 5 — re-encode luma to log for zone threshold comparisons
    luma = np.asarray(
        linear_to_log(np.maximum(luma_linear, 1e-10)), dtype=np.float64
    )

    # Step 6 — zone blending over Rec.709 base
    out_r       = R_base.copy()
    out_g       = G_base.copy()
    out_b       = B_base.copy()
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

        weight = fade_in * fade_out
        mask   = weight > best_weight

        # Blend zone color over Rec.709 base (not over gray)
        mixed_r = fr * weight + R_base * (1.0 - weight)
        mixed_g = fg * weight + G_base * (1.0 - weight)
        mixed_b = fb * weight + B_base * (1.0 - weight)

        best_weight = np.where(mask, weight,   best_weight)
        out_r       = np.where(mask, mixed_r,  out_r)
        out_g       = np.where(mask, mixed_g,  out_g)
        out_b       = np.where(mask, mixed_b,  out_b)

    out_r = np.clip(out_r, 0.0, 1.0)
    out_g = np.clip(out_g, 0.0, 1.0)
    out_b = np.clip(out_b, 0.0, 1.0)

    # Reorder: [ri, gi, bi] → [bi, gi, ri] then ravel  (.cube scan order)
    flat_r = out_r.transpose(2, 1, 0).ravel()
    flat_g = out_g.transpose(2, 1, 0).ravel()
    flat_b = out_b.transpose(2, 1, 0).ravel()

    lut = list(zip(flat_r.tolist(), flat_g.tolist(), flat_b.tolist()))
    console.print(f"  [green]✓[/green] {len(lut):,} entries generated.")
    return lut


# ───────────────────────────────────────────────────────────────────────
#  CUBE FILE WRITER
# ───────────────────────────────────────────────────────────────────────

def write_hybrid_cube(
    lut: list,
    size: int,
    filename: str,
    profile: dict,
    gamut_cs_name: str,
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
        zone_lines.append(f"#   {hex_col:<10}  {label:<14}  log {lo:.3f} – {hi:.3f}")

    header = (
        f"# Hybrid Monitor LUT\n"
        f"# Profile  : {profile['name']}\n"
        f"# Cameras  : {profile['cameras']}\n"
        f"# Input    : {profile['name']} log / {gamut_cs_name}\n"
        f"# Output   : Rec.709 base + narrow exposure tracer overlays\n"
        f"# Generator: hybrid_monitor_lut.py\n"
        f"#\n"
        f"# Base transform pipeline:\n"
        f"#   Log decode  →  {gamut_cs_name} → Rec.709 matrix  →  Rec.709 OETF\n"
        f"#\n"
        f"# Overlay zones (thin tracers over Rec.709 base):\n"
        + "\n".join(zone_lines) +
        f"\n#   All other levels → standard Rec.709 color (no tracer)\n"
        f"#\n"
        f'TITLE "HybridMonitor {profile["name"]}"\n'
        f"LUT_3D_SIZE {size}\n"
        f"DOMAIN_MIN 0.0 0.0 0.0\n"
        f"DOMAIN_MAX 1.0 1.0 1.0\n\n"
    )

    with open(filename, "w") as f:
        f.write(header)
        for (r, g, b) in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    console.print(f"  [green]✓[/green] Written to: [bold]{filename}[/bold]")


# ───────────────────────────────────────────────────────────────────────
#  GRADIENT PREVIEW IMAGE  (Pillow)
#
#  A neutral log ramp (R=G=B=t) is a good preview: it shows exactly where
#  each tracer zone activates.  Non-zone pixels render as neutral Rec.709
#  gray (since a neutral input stays neutral through any gamut matrix).
# ───────────────────────────────────────────────────────────────────────

def generate_hybrid_preview(
    parsed_zones: list,
    profile: dict,
    gamut_matrix: np.ndarray,
    output_path: str,
    width: int = 1920,
    height: int = 200,
) -> None:
    """
    Render a false-color-over-Rec.709 gradient preview PNG.

    Left = log 0 (crushed black)  →  Right = log 1 (overexposed).
    Non-tracer pixels appear as a neutral gray ramp (the Rec.709 base for
    a neutral input).  Zone pixels are tinted with the tracer color.
    """
    t = np.linspace(0.0, 1.0, width)
    row = np.zeros((width, 3), dtype=np.uint8)

    for i, v in enumerate(t):
        r_out, g_out, b_out = apply_hybrid_false_color(
            float(v), float(v), float(v), parsed_zones, profile, gamut_matrix
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

def print_hybrid_zone_table(
    zones_config: list,
    parsed_zones: list,
    profile: dict,
    gamut_matrix: np.ndarray,
) -> None:
    """
    Print a Rich table showing each zone with its terminal color swatch.

    The "Output" column shows the actual LUT output at the zone center for
    a neutral gray input.  Non-tracer output (between zones) shows what the
    Rec.709 base looks like for that tonal value.
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Zone",         style="bold",  min_width=16)
    table.add_column("Log Range",    style="dim",   min_width=15)
    table.add_column("Tracer Color", justify="center", min_width=14)
    table.add_column("LUT Output",   justify="center", min_width=14)
    table.add_column("Match",        justify="center", min_width=5)

    for (stop_val, hex_col), (lo, hi, fr, fg, fb, bw) in zip(zones_config, parsed_zones):
        label = (
            "white_clip" if stop_val == "white_clip" else
            "black_clip" if stop_val == "black_clip" else
            f"{stop_val:+.0f} stops"
        )
        center = (lo + hi) / 2.0
        r_out, g_out, b_out = apply_hybrid_false_color(
            center, center, center, parsed_zones, profile, gamut_matrix
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        match = (
            "[green]✓[/green]"
            if hex_out.upper() == hex_col.upper()
            else "[yellow]~[/yellow]"
        )
        table.add_row(label, f"{lo:.3f} – {hi:.3f}", _swatch(hex_col), _swatch(hex_out), match)

    # Show a between-zone sample — should be neutral Rec.709 (some shade of gray)
    if len(parsed_zones) >= 2:
        sorted_z = sorted(parsed_zones, key=lambda z: z[0])
        gap_mid  = (sorted_z[0][1] + sorted_z[1][0]) / 2.0
        r_out, g_out, b_out = apply_hybrid_false_color(
            gap_mid, gap_mid, gap_mid, parsed_zones, profile, gamut_matrix
        )
        hex_out = "#{:02X}{:02X}{:02X}".format(
            int(r_out * 255), int(g_out * 255), int(b_out * 255)
        )
        table.add_row(
            "[dim](between zones)[/dim]",
            f"[dim]{gap_mid:.3f}[/dim]",
            "[dim](Rec.709)[/dim]",
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
    help="Generate a hybrid monitor LUT: Rec.709 base + thin false color tracers.",
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
            help="Tracer zone half-width in stops. Keep small (0.04–0.10) for thin bands.",
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
            help="Output .cube filename. Default: luts/HybridMonitor_{profile}.cube",
        ),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(
            help="Generate a gradient preview PNG alongside the .cube file.",
        ),
    ] = True,
) -> None:
    """Generate a hybrid monitor LUT: Rec.709 base + thin false color tracers."""

    # ── Validate inputs ───────────────────────────────────────────────
    if profile not in PROFILES:
        console.print(f"[bold red]Error:[/bold red] Unknown profile [bold]'{profile}'[/bold]")
        console.print(f"[dim]Valid options: {', '.join(PROFILES.keys())}[/dim]")
        raise typer.Exit(1)

    if size < 2:
        console.print(f"[bold red]Error:[/bold red] --size must be ≥ 2, got {size}")
        raise typer.Exit(1)

    if half_width <= blend:
        console.print(
            f"[bold red]Error:[/bold red] --half-width ({half_width}) must be greater "
            f"than --blend ({blend}) to leave a solid core."
        )
        raise typer.Exit(1)

    log_profile_obj = PROFILES[profile]
    gamut_cs_name   = _GAMUT_CS_NAME[profile]
    gamut_matrix    = _GAMUT_MATRICES[profile]
    filename        = (output or OUTPUT_FILENAME).replace("{profile}", profile)

    # Blend ratio info
    blend_ratio      = blend / half_width
    solid_core_stops = 2.0 * (half_width - blend)
    total_zone_stops = len(ZONES) * 2 * half_width
    pct_rec709       = max(0.0, (1.0 - total_zone_stops / 10.0)) * 100.0

    # ── Header ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Hybrid Monitor LUT Generator[/bold]", style="dim"))
    console.print()
    console.print(f"  [dim]Profile   :[/dim]  {log_profile_obj['name']}")
    console.print(f"  [dim]Cameras   :[/dim]  [dim]{log_profile_obj['cameras']}[/dim]")
    console.print(f"  [dim]Input CS  :[/dim]  {profile.upper()} log / {gamut_cs_name}")
    console.print(f"  [dim]Output CS :[/dim]  Rec.709 + tracer overlays")
    console.print(f"  [dim]Mid gray  :[/dim]  {log_profile_obj['middle_gray']:.4f}  [dim](log-encoded 18% gray)[/dim]")
    console.print(f"  [dim]LUT size  :[/dim]  {size}³  [dim]({size**3:,} entries)[/dim]")
    console.print(f"  [dim]Output    :[/dim]  {filename}")

    if size not in (17, 33, 65):
        console.print(
            f"\n  [yellow]⚠[/yellow]  Non-standard LUT size {size}. Common NLE sizes are 17, 33, 65."
        )

    # ── Zone width analysis ───────────────────────────────────────────
    console.print()
    console.print(f"  [dim]Zone width :[/dim]  ±{half_width} stops  [dim](tracer band = {half_width*2:.2f} stops wide)[/dim]")
    console.print(f"  [dim]Blend      :[/dim]  {blend} stops  [dim](blend ratio: {blend_ratio:.0%})[/dim]")
    console.print(f"  [dim]Solid core :[/dim]  {solid_core_stops:.3f} stops per zone")

    if pct_rec709 >= 85:
        rec709_label = f"[green]{pct_rec709:.0f}% Rec.709[/green]  ✓ meets target"
    else:
        rec709_label = f"[yellow]{pct_rec709:.0f}% Rec.709[/yellow]  ⚠ below 85% target — consider narrowing --half-width"
    console.print(f"  [dim]Rec.709 %  :[/dim]  {rec709_label}")

    if blend_ratio > 0.40:
        console.print(
            f"\n  [yellow]⚠[/yellow]  Blend ratio {blend_ratio:.0%} is high. "
            f"Consider reducing [bold]--blend[/bold] to [bold]{half_width * 0.27:.3f}[/bold]."
        )

    # ── Zone preview table ────────────────────────────────────────────
    console.print()
    parsed_zones = build_zones(ZONES, log_profile_obj, half_width, blend)
    check_zone_overlaps(parsed_zones)
    print_hybrid_zone_table(ZONES, parsed_zones, log_profile_obj, gamut_matrix)

    # ── Build and write ───────────────────────────────────────────────
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    lut_data = build_hybrid_lut(parsed_zones, size, log_profile_obj, gamut_matrix)
    write_hybrid_cube(
        lut_data, size, filename, log_profile_obj,
        gamut_cs_name, ZONES, parsed_zones,
    )

    if preview:
        preview_path = os.path.splitext(filename)[0] + "_preview.png"
        generate_hybrid_preview(parsed_zones, log_profile_obj, gamut_matrix, preview_path)

    # ── Footer ────────────────────────────────────────────────────────
    size_kb = os.path.getsize(filename) / 1024
    console.print()
    console.print(
        f"[bold green]✅ Done![/bold green]  [bold]{filename}[/bold]  [dim]({size_kb:.1f} KB)[/dim]"
    )
    console.print(
        "   Load into [bold]DaVinci Resolve[/bold], [bold]Premiere Pro[/bold], "
        "[bold]FCPX[/bold], or your camera / monitor LUT slot."
    )
    console.print(Rule(style="dim"))
    console.print()


if __name__ == "__main__":
    app()
