import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.predict", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--history-csv" in result.stdout


def test_cli_does_not_ship_example_data() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "data").exists()
    assert not list(root.glob("*.csv"))


def test_cli_runs_with_temp_csv_and_mock_timesfm() -> None:
    root = Path(__file__).resolve().parents[1]
    temp_root = root / "runs" / "_pytest_cli"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    history_csv = temp_root / "history.csv"
    target_csv = temp_root / "target.csv"
    output_dir = temp_root / "run"
    _make_frame(["h1", "h2", "h3"], 35).to_csv(history_csv, index=False)
    _make_frame(["t1"], 8).to_csv(target_csv, index=False)

    env = os.environ.copy()
    env["AMGLA_MOCK_TIMESFM"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.predict",
            "--history-csv",
            str(history_csv),
            "--target-csv",
            str(target_csv),
            "--output-dir",
            str(output_dir),
            "--horizon",
            "7",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    prediction = pd.read_csv(output_dir / "prediction.csv")
    neighbors = pd.read_csv(output_dir / "neighbors.csv")
    assert prediction.shape[0] == 7
    assert neighbors.shape[0] == 2
    assert (output_dir / "config.json").exists()
    assert (output_dir / "run_summary.md").exists()
    shutil.rmtree(temp_root)


def _make_frame(products: list[str], periods: int) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=periods)
    rows = []
    for offset, product in enumerate(products):
        for i, date in enumerate(dates):
            rows.append({"product_id": product, "date": date, "sales": float(offset * 10 + i + 1)})
    return pd.DataFrame(rows)
