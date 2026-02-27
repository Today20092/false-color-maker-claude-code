"""Batch generator — writes one FalseColor LUT for every registered profile."""

import os

import false_color_lut_generator as f

os.makedirs("luts", exist_ok=True)

for key, profile in f.PROFILES.items():
    f.console.print(f"\n[bold]── Generating {profile['name']} ──[/bold]")
    filename = f"luts/FalseColor_{key}.cube"
    parsed   = f.build_zones(f.ZONES, profile, f.ZONE_HALF_WIDTH_STOPS, f.BLEND_WIDTH_STOPS)
    lut      = f.build_lut(parsed, f.LUT_SIZE, profile)   # profile now required
    f.write_cube(lut, f.LUT_SIZE, filename, profile, f.ZONES, parsed)

f.console.print("\n[bold green]✅ All profiles done[/bold green] — check the [bold]luts/[/bold] folder")
