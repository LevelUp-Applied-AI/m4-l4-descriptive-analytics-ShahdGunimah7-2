import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from eda_report import generate_eda_report


def test_generate_eda_report_with_mixed_dataframe(tmp_path):
    df = pd.DataFrame({
        "age": [20, 21, 22, None, 24],
        "gpa": [2.5, 3.0, None, 3.2, 3.8],
        "department": ["CS", "Math", "CS", "Bio", "Eng"]
    })

    output_dir = tmp_path / "report_output"
    generate_eda_report(df, output_dir=str(output_dir))

    assert (output_dir / "auto_data_profile.txt").exists()
    assert (output_dir / "age_distribution.png").exists()
    assert (output_dir / "gpa_distribution.png").exists()
    assert (output_dir / "correlation_heatmap.png").exists()
    assert (output_dir / "missing_data_visualization.png").exists()
    assert (output_dir / "outlier_summary.txt").exists()


def test_generate_eda_report_with_numeric_subset(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 100, 5],
        "y": [10, 20, 30, 40, 50],
        "label": ["a", "b", "a", "b", "c"]
    })

    output_dir = tmp_path / "subset_output"
    generate_eda_report(df, numeric_cols=["x"], output_dir=str(output_dir))

    assert (output_dir / "auto_data_profile.txt").exists()
    assert (output_dir / "x_distribution.png").exists()
    assert not (output_dir / "y_distribution.png").exists()
    assert (output_dir / "missing_data_visualization.png").exists()
    assert (output_dir / "outlier_summary.txt").exists()


def test_generate_eda_report_handles_no_missing_values(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5.0, 6.0, 7.0, 8.0],
        "c": ["x", "y", "z", "w"]
    })

    output_dir = tmp_path / "no_missing_output"
    generate_eda_report(df, output_dir=str(output_dir))

    assert (output_dir / "auto_data_profile.txt").exists()
    assert (output_dir / "a_distribution.png").exists()
    assert (output_dir / "b_distribution.png").exists()
    assert (output_dir / "correlation_heatmap.png").exists()
    assert (output_dir / "missing_data_visualization.png").exists()
    assert (output_dir / "outlier_summary.txt").exists()