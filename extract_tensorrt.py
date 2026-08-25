import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Ultralytics YOLO models to TensorRT (.engine) format"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11s.pt",
        help="Path to the trained YOLO model weights (.pt file)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target image resolution for TensorRT engine (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Static batch size for the TensorRT engine (default: 1)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Enable FP16 (half-precision) quantization (default: False)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="Enable INT8 quantization (requires --data for calibration; default: False)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML config file (required for INT8 calibration)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device to use for export (e.g., '0'; default: '0')",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT builder workspace size in GiB (default: 4.0)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic batch-size axes in the TensorRT engine (default: False)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify the ONNX graph before TensorRT compilation (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose TensorRT build logging (default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    
    print(f"\n[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    # Transfer pretrained backbone weights when training custom YAML architectures
    model_path = args.model
    if model_path.suffix.lower() in {".yaml", ".yml"}:
        model_stem = model_path.stem

        if "-custom" in model_stem:
            base_model = model_stem.split("-", maxsplit=1)[0]
            base_weight = f"{base_model}.pt"

            print(f"[INFO] Loading compatible pretrained weights from {base_weight}...")
            model.load(base_weight)

    if args.int8 and args.data is None:
        raise ValueError(
            "INT8 quantization requires a dataset YAML for calibration. "
            "Please supply --data <dataset.yaml>."
        )

    if args.int8 and args.data is not None:
        data_path = Path(args.data).resolve()
        if not data_path.exists():
            raise FileNotFoundError(
                f"Calibration dataset YAML not found at: {data_path}"
            )
    else:
        data_path = None

    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "int8": args.int8,
        "device": args.device,
        "workspace": args.workspace,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "verbose": args.verbose,
    }
    if data_path is not None:
        export_kwargs["data"] = str(data_path)

    print("[INFO] TensorRT Export Configuration")
    print(f"       Model      : {model_path}")
    print(f"       Image size : {args.imgsz}")
    print(f"       Batch size : {args.batch}")
    print(f"       Precision  : {'INT8' if args.int8 else 'FP16' if args.half else 'FP32'}")
    print(f"       Dynamic    : {args.dynamic}")
    print(f"       Workspace  : {args.workspace} GiB")
    print(f"       Device     : {args.device}")
    if data_path:
        print(f"       Calib data : {data_path}")

    print("[INFO] Starting TensorRT export ...")
    exported_path = model.export(**export_kwargs)

    engine_path = Path(exported_path) if exported_path else model_path.with_suffix(".engine")
    print(f"\n[INFO] TensorRT engine saved to: {engine_path}")
    print("[INFO] Export complete.")


if __name__ == "__main__":
    main()
