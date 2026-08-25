import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO models on UAV/Drone Detection Datasets")
    parser.add_argument("--model", "-m", type=str, default="yolo11s.yaml", help="Model checkpoint or architecture")
    parser.add_argument("--data", "-d", type=str, default="datasets/hit_uav.yaml", help="Dataset YAML config file")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", "-b", type=int, default=-1, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image resolution")
    parser.add_argument("--device", default="0", help="Computing device (e.g., '0' for GPU 0, or 'cpu')")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--project", type=str, default=None, help="Save project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify dataset configuration exists
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML configuration not found at: {data_path}")

    # Build dictionary of training arguments
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.project:
        train_kwargs["project"] = args.project
    if args.name:
        train_kwargs["name"] = args.name

    print(f"[INFO] Initializing model: {args.model}")
    model = YOLO(args.model)

    # Transfer pretrained backbone weights when training custom YAML architectures
    model_name = Path(args.model).name
    if  model_name.endswith(".yaml") and "-p2" in model_name:
        base_weight = model_name.replace("-p2.yaml", ".pt")
        print(f"[INFO] Loading pretrained backbone weights from {base_weight}...")
        try:
            model.load(base_weight)
        except Exception as e:
            print(f"[WARNING] Could not automatically load {base_weight}: {e}")

    print(f"[INFO] Starting training on dataset: {data_path}")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
