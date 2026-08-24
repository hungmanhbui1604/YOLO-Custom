"""
speed.py – Model complexity & latency profiler for YOLO models.

Two modes (automatically detected from --model extension):
  • PyTorch  (.pt)     → reports #params, GFLOPs, and PyTorch inference latency
  • TensorRT (.engine) → reports TensorRT inference latency (params/FLOPs N/A)

You can also supply both --model (PT) and --engine simultaneously to get
complexity stats from the PT file and latency from the engine in one run.

Usage examples
--------------
# Params + FLOPs + PyTorch latency only
python speed.py --model best.pt --imgsz 640

# TensorRT latency only
python speed.py --engine best.engine --imgsz 640 --batch 1

# Params/FLOPs from PT + TRT latency in one shot
python speed.py --model best.pt --engine best.engine --imgsz 640 --batch 1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile YOLO model complexity (params, FLOPs) and inference latency"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to YOLO PyTorch weights (.pt) for params/FLOPs + PT latency",
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default=None,
        help="Path to compiled TensorRT engine (.engine) for TRT latency",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size used for latency measurement (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations before timing (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed iterations for latency estimation (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index (e.g. '0') or 'cpu' (default: '0')",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_str}")
    print("[WARNING] CUDA not available – falling back to CPU.")
    return torch.device("cpu")


def make_dummy_input(batch: int, imgsz: int, device: torch.device) -> torch.Tensor:
    """Return a normalised random image tensor [B, 3, H, W] on the target device."""
    return torch.rand(batch, 3, imgsz, imgsz, device=device)


def format_number(n: float, unit: str = "") -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} G{unit}"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M{unit}"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K{unit}"
    return f"{n:.0f} {unit}".strip()


# ---------------------------------------------------------------------------
# PyTorch profiling
# ---------------------------------------------------------------------------

def profile_pytorch(model_path: Path, imgsz: int, batch: int,
                    device: torch.device, warmup: int, runs: int):
    """Compute #params, GFLOPs, and PyTorch inference latency."""
    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_num_params
    except ImportError as exc:
        raise ImportError("ultralytics is required: pip install ultralytics") from exc

    print(f"\n[INFO] Loading PyTorch model: {model_path}")
    model = YOLO(model_path)
    # Transfer pretrained backbone weights when training custom YAML architectures
    model_name = Path(model_path).name
    if  model_name.endswith(".yaml"):
        if "-custom" in model_name:
            base_weight = model_name.replace("-custom.yaml", ".pt")
        elif "-ghost-p2" in model_name:
            base_weight = model_name.replace("-ghost-p2.yaml", ".pt")
        elif "-p2" in model_name:
            base_weight = model_name.replace("-p2.yaml", ".pt")
        print(f"[INFO] Loading pretrained backbone weights from {base_weight}...")
        try:
            model.load(base_weight)
        except Exception as e:
            print(f"[WARNING] Could not automatically load {base_weight}: {e}")
    pt_model = model.model.to(device).eval()

    # ---- Params ----
    n_params = get_num_params(pt_model)

    # ---- FLOPs via thop directly ----
    # NOTE: ultralytics' get_flops() silently returns 0.0 on any internal error,
    # so we call thop directly to get accurate results and surface real errors.
    try:
        import thop
        p = next(pt_model.parameters())
        stride = max(int(pt_model.stride.max()), 32) if hasattr(pt_model, "stride") else 32
        im = torch.empty((1, p.shape[1], imgsz, imgsz), device=p.device, dtype=p.dtype)
        macs, _ = thop.profile(pt_model, inputs=[im], stride=stride, verbose=False)
        gflops = macs / 1e9 * 2  # MACs → FLOPs → GFLOPs
    except ImportError:
        gflops = float("nan")
        print("[WARNING] Could not compute FLOPs. Install 'thop': pip install thop")
    except Exception as e:
        print(f"[WARNING] thop profiling failed: {e}")
        # Fallback: plain thop without stride (less accurate but works for most models)
        try:
            import thop
            dummy = make_dummy_input(1, imgsz, device)
            macs, _ = thop.profile(pt_model, inputs=(dummy,), verbose=False)
            gflops = macs * 2 / 1e9
        except Exception as e2:
            gflops = float("nan")
            print(f"[WARNING] Fallback FLOPs computation also failed: {e2}")

    print("\n" + "=" * 55)
    print("  PyTorch Model Complexity")
    print("=" * 55)
    print(f"  Parameters : {format_number(n_params, 'params')}")
    print(f"  GFLOPs     : {gflops:.3f} GFLOPs" if not (
        isinstance(gflops, float) and gflops != gflops) else "  GFLOPs     : N/A")
    print("=" * 55)

    # ---- PyTorch latency ----
    dummy = make_dummy_input(batch, imgsz, device)
    use_cuda = device.type == "cuda"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = pt_model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = pt_model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    _print_latency_table("PyTorch", latencies, batch)
    return n_params, gflops


# ---------------------------------------------------------------------------
# TensorRT profiling
# ---------------------------------------------------------------------------

def profile_tensorrt(engine_path: Path, imgsz: int, batch: int,
                     warmup: int, runs: int):
    """Measure TensorRT inference latency using tensorrt + pycuda."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 – initialises CUDA context
    except ImportError as exc:
        raise ImportError(
            "tensorrt and pycuda are required for engine profiling.\n"
            "  pip install tensorrt pycuda"
        ) from exc

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"\n[INFO] Loading TensorRT engine: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialise TensorRT engine: {engine_path}")

    context = engine.create_execution_context()

    # ---- Allocate buffers ----
    # Collect input/output binding shapes
    input_shape = (batch, 3, imgsz, imgsz)

    bindings = []
    host_inputs, host_outputs = [], []
    device_inputs, device_outputs = [], []
    output_shapes = []

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = tuple(engine.get_binding_shape(i))

        # Handle dynamic shapes: replace -1 dims with actual batch/spatial dims
        shape = tuple(
            d if d > 0 else (batch if j == 0 else imgsz)
            for j, d in enumerate(shape)
        )

        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(i):
            host_inputs.append(host_mem)
            device_inputs.append(device_mem)
            # Fill input with random data
            np.copyto(host_mem, np.random.rand(*shape).astype(dtype).ravel())
        else:
            host_outputs.append(host_mem)
            device_outputs.append(device_mem)
            output_shapes.append(shape)

    stream = cuda.Stream()

    def infer():
        # H→D
        for hm, dm in zip(host_inputs, device_inputs):
            cuda.memcpy_htod_async(dm, hm, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # D→H
        for hm, dm in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh_async(hm, dm, stream)
        stream.synchronize()

    # Warmup
    print(f"[INFO] Warming up TensorRT engine ({warmup} iters) ...")
    for _ in range(warmup):
        infer()

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    _print_latency_table("TensorRT", latencies, batch)


# ---------------------------------------------------------------------------
# TensorRT profiling via Ultralytics (simpler, no pycuda dependency)
# ---------------------------------------------------------------------------

def profile_tensorrt_ultralytics(engine_path: Path, imgsz: int, batch: int,
                                  warmup: int, runs: int):
    """
    Fallback TRT latency measurement using Ultralytics' YOLO wrapper.
    Less low-level but works without a manual pycuda setup.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required: pip install ultralytics") from exc

    print(f"\n[INFO] Loading TensorRT engine via Ultralytics: {engine_path}")
    model = YOLO(str(engine_path))
    dummy_np = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)

    # Warmup
    print(f"[INFO] Warming up TensorRT engine ({warmup} iters) ...")
    for _ in range(warmup):
        model.predict(dummy_np, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(dummy_np, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    _print_latency_table("TensorRT (Ultralytics wrapper)", latencies, batch)


# ---------------------------------------------------------------------------
# Shared latency printer
# ---------------------------------------------------------------------------

def _print_latency_table(label: str, latencies: list, batch: int):
    arr = np.array(latencies)
    mean_ms   = arr.mean()
    std_ms    = arr.std()
    p50_ms    = np.percentile(arr, 50)
    p95_ms    = np.percentile(arr, 95)
    p99_ms    = np.percentile(arr, 99)
    fps       = batch / (mean_ms / 1000)

    print("\n" + "=" * 55)
    print(f"  Latency – {label}")
    print("=" * 55)
    print(f"  Batch size : {batch}")
    print(f"  Runs       : {len(latencies)}")
    print(f"  Mean       : {mean_ms:.2f} ms  ± {std_ms:.2f} ms")
    print(f"  P50        : {p50_ms:.2f} ms")
    print(f"  P95        : {p95_ms:.2f} ms")
    print(f"  P99        : {p99_ms:.2f} ms")
    print(f"  Throughput : {fps:.1f} FPS")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.model is None and args.engine is None:
        raise ValueError(
            "At least one of --model (.pt) or --engine (.engine) must be provided."
        )

    device = resolve_device(args.device)

    # -- PyTorch profiling (params, FLOPs, PT latency) --
    if args.model is not None:
        profile_pytorch(
            model_path=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )

    # -- TensorRT latency --
    if args.engine is not None:
        engine_path = Path(args.engine).resolve()
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        # Try low-level pycuda path first; fall back to Ultralytics wrapper
        try:
            import pycuda  # noqa: F401
            import tensorrt  # noqa: F401
            profile_tensorrt(
                engine_path=engine_path,
                imgsz=args.imgsz,
                batch=args.batch,
                warmup=args.warmup,
                runs=args.runs,
            )
        except ImportError:
            print(
                "[WARNING] pycuda/tensorrt not importable – "
                "falling back to Ultralytics wrapper for TRT latency."
            )
            profile_tensorrt_ultralytics(
                engine_path=engine_path,
                imgsz=args.imgsz,
                batch=args.batch,
                warmup=args.warmup,
                runs=args.runs,
            )

    print("\n[INFO] Profiling complete.")


if __name__ == "__main__":
    main()
