"""Reproduce demo images, numerical errors and benchmark plots for the report."""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from gaussian_filter import GaussianParams, gaussian_native, gaussian_opencv, to_uint8
from main import save_image


def make_sample(size=256):
    """Own deterministic test image: smooth gradient, edges, thin lines and noise."""
    y, x = np.mgrid[:size, :size]
    clean = np.stack((40 + 130 * x / size, 35 + 140 * y / size,
                      60 + 90 * (x + y) / (2 * size)), axis=-1).astype(np.uint8)
    cv2.rectangle(clean, (size // 9, size // 7), (size // 2, size // 2), (210, 70, 45), -1)
    cv2.circle(clean, (3 * size // 4, size // 3), size // 6, (30, 215, 240), -1)
    for offset in range(size // 10, size * 9 // 10, max(2, size // 32)):
        cv2.line(clean, (offset, size * 3 // 5), (offset, size * 4 // 5), (235, 235, 235), 1)
    cv2.putText(clean, "GAUSS", (size // 8, size * 15 // 16), cv2.FONT_HERSHEY_SIMPLEX,
                size / 300, (245, 245, 245), max(1, size // 128), cv2.LINE_AA)
    noise = np.random.default_rng(42).normal(0, 20, clean.shape)
    return clean, to_uint8(clean.astype(np.float64) + noise)


def measure(function, image, params, repeats):
    """Warm up, then batch fast calls to avoid timer-resolution dominated samples."""
    start = perf_counter_ns()
    function(image, params)
    elapsed = max(perf_counter_ns() - start, 1)
    batch = max(1, min(2000, int(20_000_000 / elapsed)))
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(batch):
            function(image, params)
        samples.append((perf_counter_ns() - start) / 1e6 / batch)
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "batch": batch}


def display(ax, image, title):
    ax.imshow(to_uint8(image)[:, :, ::-1])
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("report"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--kernels", type=int, nargs="+", default=[3, 7, 15])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 16 for size in args.sizes):
        parser.error("repeats >= 1; sizes >= 16")
    try:
        for k in args.kernels:
            GaussianParams(k, k)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    # Disable parallel regions; 0 also works with the macOS GCD backend.
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    plt.rcParams.update({"figure.dpi": 140, "font.family": "DejaVu Sans"})

    clean, noisy = make_sample()
    save_image(args.output / "clean.png", clean)
    save_image(args.output / "input.png", noisy)
    params = GaussianParams(7, 7, 1.5)
    native = gaussian_native(noisy, params)
    library = gaussian_opencv(noisy, params)
    save_image(args.output / "native.png", native)
    save_image(args.output / "opencv.png", library)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
    for ax, image, title in zip(axes, (clean, noisy, native, library),
                                ("Оригинал", "Шум: σ = 20", "Python: 7×7, σ = 1.5", "OpenCV: 7×7, σ = 1.5")):
        display(ax, image, title)
    fig.savefig(args.output / "comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    for ax, (k, sx, sy) in zip(axes.flat, ((3, 1.5, 1.5), (7, 1.5, 1.5), (15, 1.5, 1.5),
                                         (15, 0.7, 0.7), (15, 3, 3), (15, 3, 0.5))):
        display(ax, gaussian_opencv(noisy, GaussianParams(k, k, sx, sy)),
                f"{k}×{k}, σx = {sx}, σy = {sy}")
    fig.savefig(args.output / "parameters.png")
    plt.close(fig)

    rows, raw = [], []
    for size in args.sizes:
        _, image = make_sample(size)
        for kernel in args.kernels:
            params = GaussianParams(kernel, kernel, 1.5)
            timings = {name: measure(fn, image, params, args.repeats) for name, fn in
                       (("native", gaussian_native), ("opencv", gaussian_opencv))}
            error = np.abs(gaussian_native(image, params) - gaussian_opencv(image, params))
            row = dict(size=size, kernel=kernel, sigma_x=1.5, sigma_y=1.5,
                       native_ms=timings['native']['median_ms'],
                       opencv_ms=timings['opencv']['median_ms'],
                       speedup=timings['native']['median_ms'] / timings['opencv']['median_ms'],
                       max_error=float(error.max()), mean_error=float(error.mean()))
            rows.append(row)
            raw.append(dict(size=size, kernel=kernel, **timings))
            print(f"{size}×{size}, {kernel}×{kernel}: Python {row['native_ms']:.2f} ms, "
                  f"OpenCV {row['opencv_ms']:.3f} ms, ускорение {row['speedup']:.1f}×", flush=True)
    with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, platform=platform.platform(), machine=platform.machine(),
                    processor=platform.processor(), numpy=np.__version__, opencv=cv2.__version__,
                    matplotlib=matplotlib.__version__, opencv_threads=cv2.getNumThreads(),
                    opencl=cv2.ocl.useOpenCL(), input_dtype="uint8", processing_dtype="float64",
                    channels=3, seed=42, repeats=args.repeats, measurements=raw)
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for kernel in args.kernels:
        subset = [row for row in rows if row['kernel'] == kernel]
        for ax, key in zip(axes, ('native_ms', 'opencv_ms')):
            ax.plot([row['size'] for row in subset], [row[key] for row in subset],
                    marker='o', label=f"Ядро {kernel}×{kernel}")
            ax.set_xlabel("Сторона RGB-изображения, пиксели")
            ax.set_ylabel("Медианное время, мс")
            ax.grid(alpha=0.25)
            ax.legend()
    axes[0].set_title("Нативный Python")
    axes[1].set_title("OpenCV (1 поток)")
    fig.savefig(args.output / "benchmark.png")
    plt.close(fig)
    metrics = dict(max_error=float(np.abs(native - library).max()),
                   mean_error=float(np.abs(native - library).mean()),
                   noisy_mse=float(np.mean((noisy.astype(float) - clean) ** 2)),
                   filtered_mse=float(np.mean((native - clean) ** 2)))
    (args.output / "quality.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["| RGB-изображение | Ядро | Python, мс | OpenCV, мс | Ускорение | Макс. ошибка |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['size']}×{row['size']} | {row['kernel']}×{row['kernel']} | "
                     f"{row['native_ms']:.2f} | {row['opencv_ms']:.3f} | {row['speedup']:.1f}× | "
                     f"{row['max_error']:.2e} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
