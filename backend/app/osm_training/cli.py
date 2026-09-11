"""CLI for OSM golf-course ranking and segmentation training data.

Examples:

  python -m app.osm_training rank --region lisbon --out ../data/osm_training
  python -m app.osm_training rank --bbox -9.5 38.6 -9.1 38.9 --out ../data/osm_training
  python -m app.osm_training rank --global --out ../data/osm_training
  python -m app.osm_training dataset --ranking ../data/osm_training/ranking.csv --courses 200
  python -m app.osm_training dataset --ranking ../data/osm_training/ranking.csv --min-score 98 --courses 500
  python -m app.osm_training train --dataset ../data/osm_training/dataset
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import settings
from app.osm_training.census import (
    GOLF_REGIONS,
    REGION_GROUPS,
    rank_courses,
    resolve_bboxes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eagle-eye-osm-training",
        description=(
            "Rank OSM golf courses by mapping completeness and build a "
            "SegFormer/Mask2Former training set from the top candidates."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_rank_parser(sub)
    _add_dataset_parser(sub)
    _add_train_parser(sub)
    _add_regions_parser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


def _add_rank_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("rank", help="Download OSM golf courses and write a ranked CSV")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=None)
    parser.add_argument("--region", default=None, help="Named region (see `regions`)")
    parser.add_argument("--global", dest="global_scan", action="store_true", help="All golf regions")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--step", type=float, default=8.0, help="Tile size in degrees")
    parser.add_argument("--delay", type=float, default=1.0, help="Pause between Overpass tiles (s)")
    parser.set_defaults(func=_cmd_rank)


def _add_dataset_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("dataset", help="Download imagery and rasterize OSM masks for top courses")
    parser.add_argument("--ranking", type=Path, required=True, help="ranking.csv from the rank command")
    parser.add_argument("--out", type=Path, default=None, help="Dataset directory")
    parser.add_argument(
        "--courses",
        "--top",
        dest="courses",
        type=int,
        default=100,
        help="How many courses to include (highest score first). Default 100.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=100.0,
        help="Minimum completeness score. Default 100; pass 98 to include near-perfect courses.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--chip-stride", type=int, default=384)
    parser.add_argument(
        "--sources",
        default="esri,clarity",
        help="Comma-separated mosaics: esri (World Imagery) and/or clarity",
    )
    parser.set_defaults(func=_cmd_dataset)


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="Fine-tune SegFormer on the rasterized dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--no-onnx", action="store_true")
    parser.set_defaults(func=_cmd_train)


def _add_regions_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("regions", help="List built-in Overpass scan regions")
    parser.set_defaults(func=_cmd_regions)


def _cmd_rank(args: argparse.Namespace) -> int:
    out_dir = args.out or (settings.data_dir / "osm_training")
    bbox = tuple(args.bbox) if args.bbox is not None else None
    try:
        bboxes = resolve_bboxes(bbox=bbox, region=args.region, global_scan=args.global_scan)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    records = asyncio.run(
        rank_courses(
            bboxes=bboxes,
            cache_dir=out_dir / "cache",
            step_deg=args.step,
            delay_s=args.delay,
            progress=print,
            out_dir=out_dir,
        )
    )
    csv_path = out_dir / "ranking.csv"
    top = records[:10]
    print(f"Wrote {len(records)} courses → {csv_path}")
    for rank, row in enumerate(top, start=1):
        print(f"  {rank:3d}  {row.score:6.1f}  {row.name or (row.osm_type + '/' + str(row.osm_id))}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from app.osm_training.dataset import build_dataset

    out_dir = args.out or (args.ranking.parent / "dataset")
    print(
        f"Using up to {args.courses} courses with score >= {args.min_score}. "
        "Imagery will be fetched from the best configured source. "
        "Only train on imagery you have a license to use."
    )
    asyncio.run(
        build_dataset(
            ranking=args.ranking,
            out_dir=out_dir,
            top=args.courses,
            min_score=args.min_score,
            val_frac=args.val_frac,
            chip_size=args.chip_size,
            chip_stride=args.chip_stride,
            imagery_sources=tuple(
                part.strip() for part in str(args.sources).split(",") if part.strip()
            ),
            progress=print,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from app.osm_training.train import train_segformer

    out_dir = args.out or (args.dataset / "model")
    summary = train_segformer(
        dataset_dir=args.dataset,
        out_dir=out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        export_onnx=not args.no_onnx,
    )
    print(f"Checkpoint: {summary['checkpoint']}")
    if summary.get("onnx"):
        print(f"ONNX: {summary['onnx']}")
    return 0


def _cmd_regions(_args: argparse.Namespace) -> int:
    print("Regions:")
    for name, bbox in sorted(GOLF_REGIONS.items()):
        print(f"  {name:20s}  {bbox}")
    print("Groups:")
    for name, members in sorted(REGION_GROUPS.items()):
        print(f"  {name:20s}  {', '.join(members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
