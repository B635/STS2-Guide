"""Generate the original STS2 Guide tray/EXE icon deterministically."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def render_icon(size: int = 1024) -> Image.Image:
    scale = size / 256

    def pts(values):
        return tuple((round(x * scale), round(y * scale)) for x, y in values)

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (8 * scale, 8 * scale, 248 * scale, 248 * scale),
        radius=52 * scale,
        fill=(17, 25, 37, 255),
        outline=(47, 65, 84, 255),
        width=max(1, round(6 * scale)),
    )

    # Original abstract spire silhouette; no game art or official logo.
    draw.polygon(
        pts(((128, 29), (194, 207), (62, 207))),
        fill=(218, 126, 43, 255),
    )
    draw.polygon(
        pts(((128, 48), (159, 132), (97, 132))),
        fill=(250, 190, 80, 255),
    )
    draw.polygon(
        pts(((95, 137), (161, 137), (178, 184), (78, 184))),
        fill=(166, 76, 34, 255),
    )
    draw.rounded_rectangle(
        (70 * scale, 197 * scale, 186 * scale, 216 * scale),
        radius=8 * scale,
        fill=(235, 154, 52, 255),
    )

    # A cyan decision path distinguishes Guide from a generic spire mark.
    path = pts(((67, 171), (101, 153), (127, 167), (162, 112), (191, 93)))
    draw.line(
        path,
        fill=(73, 210, 220, 255),
        width=max(2, round(9 * scale)),
        joint="curve",
    )
    for x, y in ((67, 171), (127, 167), (191, 93)):
        radius = 10 * scale
        draw.ellipse(
            (
                x * scale - radius,
                y * scale - radius,
                x * scale + radius,
                y * scale + radius,
            ),
            fill=(235, 252, 250, 255),
            outline=(34, 142, 155, 255),
            width=max(1, round(4 * scale)),
        )
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "release",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    large = render_icon()
    png_path = args.output_dir / "sts2-guide.png"
    ico_path = args.output_dir / "sts2-guide.ico"
    png = large.resize((256, 256), Image.Resampling.LANCZOS)
    png.save(png_path, format="PNG", optimize=True)
    png.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (48, 48),
               (64, 64), (128, 128), (256, 256)],
    )
    print(png_path)
    print(ico_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
