"""Run the AMGLA cold-start prediction workflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from amgla import AMGLAColdStartPipeline, AMGLAConfig
from amgla.data import load_sales_csv
from amgla.timesfm_adapter import TimesFMForecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cold-start financial product sales with AMGLA.")
    parser.add_argument("--history-csv", required=True, help="CSV with complete historical products.")
    parser.add_argument("--target-csv", required=True, help="CSV with cold-start products.")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts.")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon.")
    parser.add_argument("--history-length", type=int, default=21, help="Observed context length.")
    parser.add_argument("--ca-top-k", type=int, default=2, help="Number of similar products for CA.")
    parser.add_argument("--sa-top-k", type=int, default=3, help="Number of similar products for SA.")
    parser.add_argument("--blend-weight", type=float, default=0.5, help="Weight for TimesFM forecast in final blending.")
    parser.add_argument(
        "--timesfm-checkpoint",
        default="google/timesfm-2.5-200m-pytorch",
        help="TimesFM 2.5 checkpoint name or local path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AMGLAConfig(
        history_length=args.history_length,
        forecast_horizon=args.horizon,
        ca_top_k=args.ca_top_k,
        sa_top_k=args.sa_top_k,
        blend_weight=args.blend_weight,
        timesfm_checkpoint=args.timesfm_checkpoint,
    )
    history = load_sales_csv(args.history_csv)
    target = load_sales_csv(args.target_csv)
    adapter = _MockTimesFMAdapter() if os.environ.get("AMGLA_MOCK_TIMESFM") == "1" else None
    pipeline = AMGLAColdStartPipeline(config, timesfm_adapter=adapter)
    result = pipeline.fit(history).predict(target)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_dir / "prediction.csv", index=False)
    result.neighbors.to_csv(output_dir / "neighbors.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "run_summary.md").write_text(_summary(result.metadata), encoding="utf-8")
    print(f"Wrote AMGLA predictions to {output_dir}")


def _summary(metadata: dict) -> str:
    return "\n".join(
        [
            "# AMGLA Run Summary",
            "",
            f"- Reference products: {metadata['num_reference_products']}",
            f"- Target products: {metadata['num_target_products']}",
            f"- Augmented contexts from SA: {metadata['num_augmented_contexts']}",
            f"- Representation: {metadata['representation']}",
            "",
        ]
    )


class _MockTimesFMAdapter:
    """Internal test hook used by CLI tests without downloading TimesFM."""

    def encode_hidden(self, inputs) -> np.ndarray:
        embeddings = []
        for values in inputs:
            arr = np.asarray(values, dtype=float)
            embeddings.append(
                [
                    float(arr.mean()),
                    float(arr.std()),
                    float(arr[0]),
                    float(arr[-1]),
                    float(arr[-1] - arr[0]),
                    float(arr.shape[0]),
                ]
            )
        return np.asarray(embeddings, dtype=float)

    def forecast(self, inputs, horizon: int) -> TimesFMForecast:
        point = []
        quantile = []
        for values in inputs:
            arr = np.asarray(values, dtype=float)
            slope = (arr[-1] - arr[0]) / max(arr.shape[0] - 1, 1)
            base = arr[-1] + slope * np.arange(1, horizon + 1)
            point.append(base)
            quantile.append(np.stack([base, base - 1.0, base, base + 1.0], axis=1))
        return TimesFMForecast(np.asarray(point), np.asarray(quantile))


if __name__ == "__main__":
    main()
