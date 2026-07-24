"""Excel report writer for interval-bucketed ATCC counts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.aggregator import IntervalBucket
from src.config_loader import resolve_path

logger = logging.getLogger(__name__)

COLUMNS = [
    "Interval Start",
    "Interval End",
    "Lane",
    "Direction",
    "Vehicle Class",
    "Count",
]


class ExcelExporter:
    """Write / append ATCC count rows to an ``.xlsx`` workbook."""

    def __init__(self, config: dict[str, Any], session_id: str | None = None) -> None:
        """Configure output path and write mode from export settings.

        Args:
            config: Full pipeline config.
            session_id: Optional suffix for session-mode filenames.
        """
        export_cfg = config.get("export", {})
        self.mode: str = str(export_cfg.get("mode", "append")).lower()
        self.sheet_strategy: str = str(export_cfg.get("sheet_strategy", "single")).lower()
        self.output_dir: Path = resolve_path(str(export_cfg.get("output_dir", "outputs/reports")))
        self.filename_prefix: str = str(export_cfg.get("filename_prefix", "atcc_report"))
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "session":
            self.workbook_path = self.output_dir / f"{self.filename_prefix}_{self.session_id}.xlsx"
        else:
            # append mode: stable running report per calendar day
            day = datetime.now().strftime("%Y%m%d")
            self.workbook_path = self.output_dir / f"{self.filename_prefix}_{day}.xlsx"

        logger.info("ExcelExporter ready → %s (mode=%s, sheets=%s)", self.workbook_path, self.mode, self.sheet_strategy)

    def _sheet_name(self, bucket: IntervalBucket) -> str:
        """Choose worksheet name based on sheet strategy."""
        if self.sheet_strategy == "per_day":
            return bucket.interval_start.strftime("%Y-%m-%d")
        return "counts"

    def _bucket_to_rows(self, bucket: IntervalBucket) -> list[dict[str, Any]]:
        """Flatten a bucket into row dictionaries matching ``COLUMNS``."""
        start = bucket.interval_start.isoformat()
        end = bucket.interval_end.isoformat()
        rows: list[dict[str, Any]] = []
        if not bucket.counts:
            # Emit a zero-activity marker row so the interval appears in the report.
            rows.append(
                {
                    "Interval Start": start,
                    "Interval End": end,
                    "Lane": "",
                    "Direction": "",
                    "Vehicle Class": "",
                    "Count": 0,
                }
            )
            return rows

        for (lane, direction, class_name), count in sorted(bucket.counts.items()):
            rows.append(
                {
                    "Interval Start": start,
                    "Interval End": end,
                    "Lane": lane,
                    "Direction": direction,
                    "Vehicle Class": class_name,
                    "Count": int(count),
                }
            )
        return rows

    def write_bucket(self, bucket: IntervalBucket) -> Path:
        """Append (or create) rows for a completed interval bucket.

        Args:
            bucket: Aggregated counts for one time window.

        Returns:
            Path to the workbook written.
        """
        new_df = pd.DataFrame(self._bucket_to_rows(bucket), columns=COLUMNS)
        sheet = self._sheet_name(bucket)

        if self.workbook_path.is_file():
            existing = pd.read_excel(self.workbook_path, sheet_name=None, engine="openpyxl")
            if sheet in existing:
                combined = pd.concat([existing[sheet], new_df], ignore_index=True)
            else:
                combined = new_df
            with pd.ExcelWriter(self.workbook_path, engine="openpyxl") as writer:
                for name, df in existing.items():
                    if name == sheet:
                        continue
                    df.to_excel(writer, sheet_name=name, index=False)
                combined.to_excel(writer, sheet_name=sheet, index=False)
        else:
            with pd.ExcelWriter(self.workbook_path, engine="openpyxl") as writer:
                new_df.to_excel(writer, sheet_name=sheet, index=False)

        logger.info("Wrote %d rows to %s [%s]", len(new_df), self.workbook_path, sheet)
        return self.workbook_path
