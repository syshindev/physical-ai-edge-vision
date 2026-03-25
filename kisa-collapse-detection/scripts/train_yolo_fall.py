"""YOLO11x fine-tuning - enhance fallen person detection
- Based on existing yolo11x.pt
- Low lr + short epochs + freeze backbone -> maintain generalization
- Use --server flag when running on training server (relative paths)
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLO11x fall detection fine-tuning")
    parser.add_argument("--server", action="store_true",
                        help="Use relative paths for training server")
    args = parser.parse_args()

    if args.server:
        data_yaml = Path(__file__).parent / "data.yaml"
        output_dir = Path(__file__).parent / "output"
    else:
        data_yaml = PROJECT_ROOT / "datasets" / "fall_finetune" / "data.yaml"
        output_dir = PROJECT_ROOT / "weights" / "yolo11x_fall"

    model = YOLO("yolo11x.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=960,
        batch=4,
        lr0=0.0005,         # Low learning rate (maintain generalization)
        lrf=0.01,           # Final lr = lr0 * lrf
        freeze=10,          # Freeze first 10 backbone layers
        patience=5,         # early stopping
        save=True,
        save_period=5,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        device=0,
        workers=4,
        cos_lr=True,
        warmup_epochs=2,
        close_mosaic=5,     # Disable mosaic for last 5 epochs for stabilization
        single_cls=True,    # Single class: person
    )

    print(f"\nTraining complete!")
    print(f"best.pt: {output_dir}/train/weights/best.pt")


if __name__ == "__main__":
    main()
