import pandas as pd
import numpy as np
from config import GLOBAL_CONFIG
import matplotlib.pyplot as plt



def plot_steering_over_time(dataset: pd.DataFrame) -> None:
    groups = sorted(dataset["group_id"].dropna().unique())

    for group in groups:
        group_data = dataset[
            dataset["group_id"] == group
        ].sort_values("timestamp")

        plt.figure(figsize=(14, 5))

        plt.plot(
            group_data["timestamp"],
            group_data["steer"],
            linewidth=0.5,
        )

        plt.xlabel("Time")
        plt.ylabel("Steering")
        plt.title(f"Steering Over Time - Group {int(group)}")
        plt.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()
def plot_steering_histogram_zoom(dataset: pd.DataFrame) -> None:
    groups = sorted(dataset["group_id"].dropna().unique())

    plt.figure(figsize=(12, 6))

    for group in groups:
        group_data = dataset.loc[
            dataset["group_id"] == group,
            "steer"
        ]

        plt.hist(
            group_data,
            bins=100,
            alpha=0.5,
            label=f"Group {int(group)}",
            density=True,
        )

    plt.xlim(-0.15, 0.15)

    plt.xlabel("Steering")
    plt.ylabel("Density")
    plt.title("Steering Distribution by Group - Zoomed")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_steering_histogram(dataset: pd.DataFrame) -> None:
    groups = sorted(dataset["group_id"].dropna().unique())

    plt.figure(figsize=(12, 6))

    for group in groups:
        group_data = dataset.loc[
            dataset["group_id"] == group,
            "steer"
        ]

        plt.hist(
            group_data,
            bins=100,
            alpha=0.5,
            label=f"Group {int(group)}",
            density=True,
        )

    plt.xlabel("Steering")
    plt.ylabel("Density")
    plt.title("Steering Distribution by Group")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

def test():
    DATASET_PATH = GLOBAL_CONFIG.pipeline.csv_path

    dataset = pd.read_csv(DATASET_PATH)
    total_rows = len(dataset)

    print(f"Total number of entries in the original dataset: {total_rows}")

    dataset["timestamp"] = pd.to_datetime(
        dataset["timestamp"],
        unit="s",
    )
    dataset["dt"] = (
        dataset
        .groupby("vehicle_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    print("\n===== DATASET OVERVIEW =====")
    print(f"Samples:       {len(dataset):,}")
    print(f"Vehicles:      {dataset['vehicle_id'].nunique():,}")
    print(f"Columns:       {len(dataset.columns):,}")

    print("\n===== COLUMNS =====")
    print(dataset.columns.tolist())

    print("\n===== MISSING VALUES =====")
    print(dataset.isnull().sum().to_string())

    print("\n===== VEHICLES =====")
    vehicle_counts = (
        dataset["vehicle_id"]
        .value_counts()
        .sort_index()
    )

    print(vehicle_counts.to_string())

    print("\n===== VEHICLE STATISTICS =====")

    vehicle_stats = (
        dataset
        .groupby("vehicle_id")
        .agg(
            samples=("timestamp", "size"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        )
        .sort_values("start")
    )

    vehicle_stats["duration_min"] = (
        vehicle_stats["end"] - vehicle_stats["start"]
    ).dt.total_seconds() / 60.0

    print(vehicle_stats.to_string())

    print("\n===== STEERING DISTRIBUTION =====")
    print(dataset["steer"].describe())

    print("\n===== STEERING QUANTILES =====")
    print(
        dataset["steer"]
        .quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .to_string()
    )
    print("\n--- Largest gaps per vehicle ---")
    print(
        dataset.nlargest(20, "dt")[
            ["vehicle_id", "timestamp", "dt"]
        ].to_string(index=False)
    )

    print("\n===== VEHICLE GROUP TIMELINE =====")

    vehicle_first = (
        dataset
        .groupby("vehicle_id")["timestamp"]
        .agg(["min", "max", "count"])
        .sort_values("min")
    )

    print(vehicle_first.to_string())

    print("\n===== VEHICLE START ORDER =====")

    print(
        dataset[
            ["vehicle_id", "timestamp"]
        ]
        .sort_values("timestamp")
        .head(100)
        .to_string(index=False)
    )

    print("\n===== VEHICLE GROUPS =====")

    vehicle_ids = sorted(dataset["vehicle_id"].unique())

    for i in range(0, len(vehicle_ids), 20):
        group = vehicle_ids[i:i + 20]

        print(
            f"Group {i // 20 + 1}: "
            f"{group[0]} - {group[-1]}"
        )

        group_data = dataset[
            dataset["vehicle_id"].isin(group)
        ]

        print(
            f"  Start: {group_data['timestamp'].min()}"
        )
        print(
            f"  End:   {group_data['timestamp'].max()}"
        )
        print(
            f"  Samples: {len(group_data):,}"
    )

    print("\n===== GROUP STEERING STATISTICS =====")

    dataset["group_id"] = pd.cut(
        dataset["vehicle_id"],
        bins=[196, 235, 386, 511, 674],
        labels=[1, 2, 3, 4],
    )

    group_stats = (
        dataset
        .groupby("group_id", observed=True)
        .agg(
            samples=("steer", "size"),
            steer_mean=("steer", "mean"),
            steer_std=("steer", "std"),
            steer_min=("steer", "min"),
            steer_max=("steer", "max"),
            lane_offset_mean=("lane_offset", "mean"),
            lane_offset_std=("lane_offset", "std"),
            heading_error_mean=("heading_error", "mean"),
            heading_error_std=("heading_error", "std"),
        )
    )

    print(group_stats.to_string())



    bins = [
        -0.8,
        -0.3,
        -0.1,
        -0.05,
        -0.01,
        0.01,
        0.05,
        0.1,
        0.3,
        0.8,
    ]

    labels = [
        "extreme_left",
        "strong_left",
        "medium_left",
        "small_left",
        "deadzone",
        "small_right",
        "medium_right",
        "strong_right",
        "extreme_right",
    ]

    dataset["steer_bin"] = pd.cut(
        dataset["steer"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distribution = pd.crosstab(
        dataset["group_id"],
        dataset["steer_bin"],
        normalize="index",
    ) * 100

    print("\n===== STEERING DISTRIBUTION BY GROUP (%) =====")
    print(distribution.round(2).to_string())

    plot_steering_histogram(dataset)
    plot_steering_histogram_zoom(dataset)
    plot_steering_over_time(dataset)

if __name__ == "__main__":
    test()
