#!/usr/bin/env python3
"""Render polished demo media from the actual deterministic workflow receipt."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo"
WIDTH, HEIGHT, FPS = 960, 540, 10
SOURCE_INPUTS = [
    "ml_workflow.py",
    "requirements.txt",
    "scripts/generate_demo.py",
    "scripts/verify_demo.py",
    "tests/test_ml_workflow.py",
]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, color: str, bold: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def base(step: str, progress: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07111f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), 24, fill="#0c1b2d", outline="#203b56", width=2)
    text(draw, (56, 47), "AI / ML COURSE LAB", 17, "#7dd3fc", True)
    text(draw, (WIDTH - 205, 49), "SYNTHETIC FIXTURE", 14, "#fbbf24", True)
    draw.rounded_rectangle((56, HEIGHT - 55, WIDTH - 56, HEIGHT - 45), 5, fill="#172b40")
    draw.rounded_rectangle((56, HEIGHT - 55, 56 + int((WIDTH - 112) * progress), HEIGHT - 45), 5, fill="#22d3ee")
    text(draw, (56, HEIGHT - 83), step, 15, "#93a4b8")
    return image, draw


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, accent: str = "#22d3ee") -> None:
    draw.rounded_rectangle(box, 16, fill="#10243a", outline="#274662", width=2)
    x1, y1, _, _ = box
    text(draw, (x1 + 22, y1 + 20), title.upper(), 14, "#93a4b8", True)
    text(draw, (x1 + 22, y1 + 54), value, 27, accent, True)


def make_frame(report: dict, frame_index: int, total: int) -> Image.Image:
    seconds = frame_index / FPS
    progress = (frame_index + 1) / total
    if seconds < 2.2:
        image, draw = base("01 / RUN REAL WORKFLOW", progress)
        text(draw, (56, 140), "Leakage-safe", 49, "#f8fafc", True)
        text(draw, (56, 198), "model selection", 49, "#f8fafc", True)
        text(draw, (59, 276), "Offline • deterministic • reproducible", 23, "#7dd3fc")
        card(draw, (560, 137, 862, 238), "Seed", str(report["seed"]), "#fbbf24")
        card(draw, (560, 258, 862, 359), "Rows", str(report["rows"]), "#34d399")
        return image
    if seconds < 5.0:
        image, draw = base("02 / LOCK HOLDOUT BEFORE SELECTION", progress)
        text(draw, (56, 104), "One split. No peeking.", 34, "#f8fafc", True)
        card(draw, (56, 174, 408, 296), "Training rows", str(report["train_rows"]), "#22d3ee")
        card(draw, (432, 174, 784, 296), "Locked holdout", str(report["holdout_rows"]), "#fbbf24")
        text(draw, (56, 337), "DISJOINT SPLIT", 15, "#93a4b8", True)
        draw.rounded_rectangle((56, 371, 220, 416), 12, fill="#0f766e")
        text(draw, (99, 381), "✓  PASS", 19, "#ecfeff", True)
        text(draw, (252, 380), "Holdout excluded from CV and winner selection", 17, "#cbd5e1")
        return image
    if seconds < 10.2:
        image, draw = base("03 / TRAINING-ONLY 5-FOLD OUT-OF-FOLD SCORES", progress)
        text(draw, (56, 97), "Compare candidates", 32, "#f8fafc", True)
        best = max(item["cv_rmse"] for item in report["candidates"])
        y = 161
        for item in report["candidates"]:
            winner = item["name"] == report["winner"]
            color = "#22d3ee" if winner else "#4b647d"
            label = item["name"].replace("_", " ")
            text(draw, (56, y), label, 17, "#e2e8f0", winner)
            draw.rounded_rectangle((252, y + 2, 740, y + 24), 8, fill="#172b40")
            bar = int(488 * (item["cv_rmse"] / best))
            draw.rounded_rectangle((252, y + 2, 252 + bar, y + 24), 8, fill=color)
            text(draw, (760, y), f"{item['cv_rmse']:.3f}", 17, "#f8fafc", True)
            y += 62
        text(draw, (760, 128), "RMSE", 13, "#93a4b8", True)
        return image
    if seconds < 14.0:
        image, draw = base("04 / FIT WINNER, EVALUATE ONCE", progress)
        text(draw, (56, 102), "Winner", 17, "#93a4b8", True)
        text(draw, (56, 133), report["winner"].replace("_", " "), 38, "#22d3ee", True)
        holdout = report["holdout"]
        card(draw, (56, 218, 301, 337), "Holdout RMSE", f"{holdout['rmse']:.3f}", "#f8fafc")
        card(draw, (326, 218, 571, 337), "Holdout MAE", f"{holdout['mae']:.3f}", "#f8fafc")
        card(draw, (596, 218, 841, 337), "Holdout R²", f"{holdout['r2']:.3f}", "#34d399")
        text(draw, (56, 384), "Observed only on the fictional fixture at seed 8257", 18, "#cbd5e1")
        return image
    image, draw = base("05 / VERIFIED RECEIPT", progress)
    text(draw, (56, 103), "What this proves", 34, "#f8fafc", True)
    bullets = [
        ("✓", "Deterministic workflow runs end to end"),
        ("✓", "Preprocessing stays inside each CV fold"),
        ("✓", "Final holdout stays outside model selection"),
    ]
    y = 172
    for mark, label in bullets:
        text(draw, (64, y), mark, 24, "#34d399", True)
        text(draw, (105, y + 1), label, 20, "#e2e8f0")
        y += 57
    draw.rounded_rectangle((56, 359, 850, 419), 14, fill="#3b2411", outline="#7c4a16", width=2)
    text(draw, (78, 378), "No real-world accuracy, scale, deployment, or performance claim", 17, "#fde68a", True)
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_probe(path: Path) -> dict:
    raw = run(
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    )
    return json.loads(raw)


def main() -> int:
    if run("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to record from a dirty tracked source tree")
    source_commit = run("git", "rev-parse", "HEAD")
    source_tree = run("git", "rev-parse", "HEAD^{tree}")
    report_path = OUTPUT / "workflow-receipt.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "ml_workflow.py", "--seed", "8257", "--output", str(report_path)],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(report_path.read_text())
    duration_seconds = 17.0
    total_frames = int(duration_seconds * FPS)
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        for index in range(total_frames):
            make_frame(report, index, total_frames).save(frame_dir / f"frame-{index:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", str(frame_dir / "frame-%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(OUTPUT / "model-selection-demo.mp4"),
            ],
            check=True,
        )
        scene_starts = [0, 22, 50, 102, 140]
        scene_durations = [2200, 2800, 5200, 3800, 3000]
        images = [
            Image.open(frame_dir / f"frame-{index:04d}.png").convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=96
            )
            for index in scene_starts
        ]
        images[0].save(
            OUTPUT / "model-selection-demo.gif", save_all=True, append_images=images[1:],
            duration=scene_durations, loop=0, optimize=True, disposal=2,
        )
        make_frame(report, int(12.0 * FPS), total_frames).save(OUTPUT / "poster.png")

    artifacts = {}
    for name in ["model-selection-demo.gif", "model-selection-demo.mp4", "poster.png", "workflow-receipt.json"]:
        path = OUTPUT / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    artifacts["model-selection-demo.gif"]["probe"] = media_probe(OUTPUT / "model-selection-demo.gif")
    artifacts["model-selection-demo.mp4"]["probe"] = media_probe(OUTPUT / "model-selection-demo.mp4")
    source_blobs = {path: run("git", "rev-parse", f"HEAD:{path}") for path in SOURCE_INPUTS}
    capture = {
        "schema_version": 1,
        "scope": "deterministic synthetic fixture; no real-world accuracy, scale, deployment, or performance claim",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_blobs": source_blobs,
        "command": "python scripts/generate_demo.py",
        "seed": 8257,
        "artifacts": artifacts,
    }
    (OUTPUT / "capture.json").write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    print(f"recorded {total_frames} frames from {source_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
