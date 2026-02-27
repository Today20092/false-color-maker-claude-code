import os
import false_color_lut_generator as f

os.makedirs("luts", exist_ok=True)

for key, profile in f.PROFILES.items():
    print(f"\n── Generating {profile['name']} ──")
    filename = f"luts/FalseColor_{key}.cube"
    parsed   = f.build_zones(f.ZONES, profile, f.ZONE_HALF_WIDTH_STOPS, f.BLEND_WIDTH_STOPS)
    lut      = f.build_lut(parsed, f.LUT_SIZE)
    f.write_cube(lut, f.LUT_SIZE, filename, profile, f.ZONES, parsed)

print("\n✅ All profiles done — check the luts/ folder")
