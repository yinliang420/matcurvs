"""One-image connectivity probe for each of the 4 VLM backends.

Loads .env then calls each backend on a single label/ image and prints
(status, latency, output_tokens, n_trails). Used to validate API keys /
gateway routing before launching a full sweep.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))


def _load_env(path: Path) -> None:
    import os
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()


_load_env(KMEANS_ROOT / ".env")

from matdeplot.baselines.vlm_baseline import (  # noqa: E402
    call_vlm,
    default_backends,
)

LABEL_DIR = KMEANS_ROOT / "label"


def main() -> None:
    # pick a small simple image for the probe
    imgs = sorted(LABEL_DIR.glob("*.jpg"))
    if not imgs:
        print("no images in label/")
        return
    test_img = imgs[0]
    image_id = test_img.stem
    print(f"probe image: {image_id} ({test_img})")

    for be in default_backends():
        print(f"\n--- {be.name} ({be.backend}) @ {be.endpoint}")
        t0 = time.monotonic()
        try:
            r = call_vlm(str(test_img), image_id=image_id, backend=be)
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            continue
        dt = time.monotonic() - t0
        print(f"  success={r.success} latency={dt:.1f}s "
              f"in_tok={r.input_tokens} out_tok={r.output_tokens} "
              f"n_trails={len(r.trails)}")
        if r.error:
            print(f"  ERROR: {r.error[:300]}")
        if r.trails:
            print(f"  first trail: id={r.trails[0].line_id} "
                  f"n_pts={len(r.trails[0].points)} "
                  f"sample={r.trails[0].points[:2]}")


if __name__ == "__main__":
    main()
