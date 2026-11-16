"""Train a lightweight YOLOv8-Pose 4-corner guitar neck detector and export to ONNX.

Usage:
    1. Calibrate your guitar inside FretVision and click "Save Neck Sample" across a few angles/lighting.
    2. Install training dependency: `uv pip install ultralytics`
    3. Run: `.venv\\Scripts\\python.exe tools\\train_neck_pose.py --epochs 40`
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_dataset_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "neck_pose.yaml"
    yaml_content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "kpt_shape: [4, 3]\n"
        "names:\n"
        "  0: guitar_fretboard\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8n-Pose neck keypoint detector.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "dataset" / "neck_pose",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "models" / "fretboard_pose.onnx",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA GPU index (0 for RTX 4060) or 'cpu'")
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        print(f"No training frames found in {images_dir}. Use 'Save Neck Sample' in FretVision first.")
        return 1

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Install ultralytics first: uv pip install ultralytics")
        return 1

    yaml_path = build_dataset_yaml(args.dataset)
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.dataset / "runs"),
        name="fretboard_pose",
    )
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(onnx_path.read_bytes())
    print(f"Exported ONNX neck pose model to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
