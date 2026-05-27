from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import materialize_preprocessed_cache, write_manifest_bundle
from data.scan_dataset import scan_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CAD sequence RAG manifests.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--add-stop-samples", action="store_true")
    parser.add_argument("--no-validate-images", action="store_true")
    parser.add_argument("--stop-image-policy", default="copy_last_depth", choices=["copy_last_depth", "blank"])
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None, help="Optionally write resized/padded images and use cached paths.")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    result = scan_dataset(
        args.dataset_root,
        add_stop_samples=args.add_stop_samples,
        stop_image_policy=args.stop_image_policy,
        validate_images=not args.no_validate_images,
    )
    if args.cache_dir:
        result.samples = materialize_preprocessed_cache(result.samples, args.cache_dir, image_size=args.image_size)
    paths = write_manifest_bundle(
        result,
        args.manifest_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by_part_id=True,
        seed=args.seed,
    )
    print(json.dumps({"paths": {k: str(v) for k, v in paths.items()}, "stats": result.stats}, indent=2))


if __name__ == "__main__":
    main()
