"""OSM completeness ranking and SegFormer training-dataset pipeline.

This package is intentionally separate from the live course-vectorization job.
Rerun `rank` as OSM improves to refresh the candidate list.
"""

from app.osm_training.census import rank_courses, write_ranking_csv
from app.osm_training.scoring import FeatureCounts, completeness_score, counts_from_layers

__all__ = [
    "FeatureCounts",
    "completeness_score",
    "counts_from_layers",
    "rank_courses",
    "write_ranking_csv",
]
