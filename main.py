"""Command-line application: python main.py input.png --help."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from gaussian_filter import GaussianParams, gaussian_native, gaussian_opencv, to_uint8


def read_image(path, grayscale=False):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, to_uint8(image))
    if not ok:
        raise ValueError(f"Не удалось сохранить изображение: {path}")
    data.tofile(path)


def main():
    parser = argparse.ArgumentParser(description="ЛР1, вариант 3: усредняющий фильтр Гаусса")
    parser.add_argument("input", type=Path, help="Входное изображение")
    parser.add_argument("--kernel", nargs=2, type=int, default=[5, 5], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--sigma-x", type=float, default=1.0)
    parser.add_argument("--sigma-y", type=float, help="По умолчанию равно sigma-x")
    parser.add_argument("--method", choices=["native", "opencv", "both"], default="both")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Папка результатов")
    args = parser.parse_args()
    try:
        params = GaussianParams(*args.kernel, args.sigma_x, args.sigma_y)
        image = read_image(args.input, args.grayscale)
        results, stats = {}, {"parameters": vars(params), "shape": list(image.shape)}
        for name, function in (("native", gaussian_native), ("opencv", gaussian_opencv)):
            if args.method in (name, "both"):
                start = perf_counter()
                results[name] = function(image, params)
                stats[name + "_ms"] = (perf_counter() - start) * 1000
                save_image(args.output / (name + ".png"), results[name])
        if len(results) == 2:
            diff = np.abs(results["native"] - results["opencv"])
            stats["max_absolute_error"] = float(diff.max())
            stats["mean_absolute_error"] = float(diff.mean())
            # A black image means equality at display precision; raw error is in JSON.
            save_image(args.output / "difference.png", diff)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(2, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
