#!/usr/bin/env python3
"""Save a simple learning curve for the C bagging executable.

The script intentionally uses only the Python standard library. It runs
robot_dt with several ensemble sizes, saves numeric results to CSV, and writes
an SVG plot that can be opened in a browser or inserted into a report.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "robot_dt"
OUT_DIR = ROOT / "learning_curve"
CLASS_DATA = ROOT / "data" / "robot.csv"
REG_DATA = ROOT / "data" / "robot_runtime.csv"


def run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout


def parse_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Cannot parse metric from output:\n{text}")
    return float(match.group(1))


def save_svg(rows: list[dict[str, float]], metric: str, out_path: Path) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 30, 30, 60
    xs = [r["trees"] for r in rows]
    ys = [r[metric] for r in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
        min_y -= 1.0

    def px(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * (width - left - right)

    def py(y: float) -> float:
        return height - bottom - (y - min_y) / (max_y - min_y) * (height - top - bottom)

    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#1f77b4" />'
        for x, y in zip(xs, ys)
    )
    labels = "\n".join(
        f'<text x="{px(x):.1f}" y="{height - 25}" text-anchor="middle" font-size="12">{int(x)}</text>'
        for x in xs
    )
    title = "Learning curve: accuracy" if metric == "accuracy" else "Learning curve: RMSE"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>
{circles}
{labels}
<text x="{width/2}" y="{height-5}" text-anchor="middle" font-family="Arial" font-size="13">number of trees</text>
<text x="16" y="{height/2}" transform="rotate(-90 16,{height/2})" text-anchor="middle" font-family="Arial" font-size="13">{metric}</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "classification"
    trees = [1, 2, 3, 5, 10, 20, 30, 50]
    OUT_DIR.mkdir(exist_ok=True)

    rows: list[dict[str, float]] = []
    if mode == "regression":
        for n in trees:
            out = run([str(EXE), "train-reg", str(REG_DATA), str(n), "6", str(n)])
            rmse = parse_float(r"RMSE.*:\s*([0-9.]+)", out)
            rows.append({"trees": float(n), "rmse": rmse})
        csv_path = OUT_DIR / "regression_learning_curve.csv"
        svg_path = OUT_DIR / "regression_learning_curve.svg"
        metric = "rmse"
    else:
        for n in trees:
            out = run([str(EXE), "train", str(CLASS_DATA), str(n), "5", "entropy", str(n)])
            acc = parse_float(r"Точность.*:\s*([0-9.]+)%", out)
            rows.append({"trees": float(n), "accuracy": acc})
        csv_path = OUT_DIR / "classification_learning_curve.csv"
        svg_path = OUT_DIR / "classification_learning_curve.svg"
        metric = "accuracy"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trees", metric])
        writer.writeheader()
        writer.writerows(rows)
    save_svg(rows, metric, svg_path)
    print(f"Saved {csv_path}")
    print(f"Saved {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
