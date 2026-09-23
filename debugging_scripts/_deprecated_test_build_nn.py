import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from config import GLOBAL_CONFIG
import tensorflow as tf

FEATURES = [
    "speed",
    "acceleration",
    "distance",
    "lane_offset",
    "heading_error",
]

TARGET = "steer"

def build_steering_mlp(
    input_dim: int,
) -> tf.keras.Model:

    inputs = tf.keras.Input(
        shape=(input_dim,),
        name="features",
    )

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
    )(inputs)

    x = tf.keras.layers.Dense(
        16,
        activation="relu",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="steer",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="steering_mlp",
    )

    return model


def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
) -> tf.data.Dataset:

    dataset = tf.data.Dataset.from_tensor_slices(
        (X, y)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=100_000,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tf.keras.Model:

    train_dataset = create_tf_dataset(
        X_train,
        y_train,
        batch_size=256,
        shuffle=True,
    )

    validation_dataset = create_tf_dataset(
        X_validation,
        y_validation,
        batch_size=256,
        shuffle=False,
    )

    model = build_steering_mlp(
        input_dim=len(FEATURES),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    print("\n===== MODEL =====")
    model.summary()

    print("\n===== TRAINING =====")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        verbose=1,
    )

    return model


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:

    test_dataset = create_tf_dataset(
        X_test,
        y_test,
        batch_size=256,
        shuffle=False,
    )

    print("\n===== TEST EVALUATION =====")

    results = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True,
    )

    print(
        f"Test MSE: {results['loss']:.6f}"
    )

    print(
        f"Test MAE: {results['mae']:.6f}"
    )

    print(
        f"Zero baseline MAE: "
        f"{np.mean(np.abs(y_test)):.6f}"
    )


def analyze_model_predictions(
    model,
    X_test,
    y_test,
) -> None:

    print("\n===== MODEL PREDICTION ANALYSIS =====")

    # Predictions
    y_pred = model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    # ---------------------------------------------------------
    # Overall statistics
    # ---------------------------------------------------------

    print("\n--- OVERALL ---")

    print(
        f"Actual mean:      {y_test.mean():.6f}"
    )

    print(
        f"Prediction mean:  {y_pred.mean():.6f}"
    )

    print(
        f"Actual std:       {y_test.std():.6f}"
    )

    print(
        f"Prediction std:   {y_pred.std():.6f}"
    )

    print(
        f"Actual min:       {y_test.min():.6f}"
    )

    print(
        f"Prediction min:   {y_pred.min():.6f}"
    )

    print(
        f"Actual max:       {y_test.max():.6f}"
    )

    print(
        f"Prediction max:   {y_pred.max():.6f}"
    )

    # ---------------------------------------------------------
    # Scatter plot
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_test,
        y_pred,
        s=1,
        alpha=0.1,
    )

    # Ideal prediction line
    min_value = min(
        y_test.min(),
        y_pred.min(),
    )

    max_value = max(
        y_test.max(),
        y_pred.max(),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linewidth=2,
    )

    plt.xlabel("Actual Steering")
    plt.ylabel("Predicted Steering")
    plt.title("Actual vs Predicted Steering")

    plt.grid(
        True,
        alpha=0.2,
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Error by steering magnitude
    # ---------------------------------------------------------

    print(
        "\n===== MAE BY STEERING MAGNITUDE ====="
    )

    abs_actual = np.abs(y_test)

    bins = [
        0.0,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.4,
        np.inf,
    ]

    labels = [
        "0-0.005",
        "0.005-0.01",
        "0.01-0.025",
        "0.025-0.05",
        "0.05-0.1",
        "0.1-0.2",
        "0.2-0.4",
        ">0.4",
    ]

    categories = pd.cut(
        abs_actual,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = []

    for label in labels:

        mask = categories == label

        if mask.sum() == 0:
            continue

        actual = y_test[mask]
        predicted = y_pred[mask]

        model_mae = np.mean(
            np.abs(actual - predicted)
        )

        baseline_mae = np.mean(
            np.abs(actual)
        )

        result.append(
            {
                "steer_bin": label,
                "samples": mask.sum(),
                "percentage": (
                    mask.sum()
                    / len(y_test)
                    * 100
                ),
                "baseline_mae": baseline_mae,
                "model_mae": model_mae,
                "improvement_%": (
                    1
                    - model_mae / baseline_mae
                )
                * 100,
            }
        )

    result = pd.DataFrame(result)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    correlation = np.corrcoef(
        y_test,
        y_pred,
    )[0, 1]

    print(
        "\nPrediction correlation: "
        f"{correlation:.6f}"
    )


def analyze_steering_predictions(y_true, y_pred):
    """
    Analyze steering predictions against ground-truth steering values.

    Reports:
    - overall statistics
    - direction accuracy for different steering magnitudes
    - MAE for different steering magnitudes
    - left/right direction accuracy
    - direction confusion
    - strong-steering performance
    - examples of strong steering predictions

    Parameters
    ----------
    y_true : array-like
        Ground-truth steering values.
    y_pred : array-like
        Model-predicted steering values.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{len(y_true)} != {len(y_pred)}"
        )

    print("\n")
    print("=" * 70)
    print("STEERING PREDICTION DIAGNOSTICS")
    print("=" * 70)

    # ============================================================
    # OVERALL
    # ============================================================

    print("\n===== OVERALL =====")

    print(f"Actual mean:       {np.mean(y_true):.6f}")
    print(f"Prediction mean:   {np.mean(y_pred):.6f}")

    print(f"Actual std:        {np.std(y_true):.6f}")
    print(f"Prediction std:    {np.std(y_pred):.6f}")

    print(f"Actual min:        {np.min(y_true):.6f}")
    print(f"Prediction min:    {np.min(y_pred):.6f}")

    print(f"Actual max:        {np.max(y_true):.6f}")
    print(f"Prediction max:    {np.max(y_pred):.6f}")

    # ============================================================
    # DIRECTION ACCURACY
    # ============================================================

    print("\n===== DIRECTION ACCURACY =====")

    thresholds = [0.005, 0.01, 0.05, 0.1, 0.2]

    for threshold in thresholds:

        mask = np.abs(y_true) >= threshold

        actual = y_true[mask]
        prediction = y_pred[mask]

        if len(actual) == 0:
            continue

        correct_direction = (actual * prediction) > 0

        direction_accuracy = (
            np.mean(correct_direction) * 100
        )

        print(
            f"|steer| >= {threshold:<5} : "
            f"{direction_accuracy:6.2f}% "
            f"({len(actual):,} samples)"
        )

    # ============================================================
    # MAE BY STEERING MAGNITUDE
    # ============================================================

    print("\n===== MAE FOR STEERING MAGNITUDE =====")

    for threshold in thresholds:

        mask = np.abs(y_true) >= threshold

        actual = y_true[mask]
        prediction = y_pred[mask]

        if len(actual) == 0:
            continue

        model_mae = np.mean(
            np.abs(actual - prediction)
        )

        baseline_mae = np.mean(
            np.abs(actual)
        )

        improvement = (
            (1 - model_mae / baseline_mae) * 100
        )

        print(
            f"|steer| >= {threshold:<5} : "
            f"samples={len(actual):7,}  "
            f"model_MAE={model_mae:.6f}  "
            f"baseline_MAE={baseline_mae:.6f}  "
            f"improvement={improvement:7.2f}%"
        )

    # ============================================================
    # LEFT / RIGHT ACCURACY
    # ============================================================

    print("\n===== LEFT / RIGHT ACCURACY =====")

    left_mask = y_true < -0.05
    right_mask = y_true > 0.05

    for name, mask in [
        ("LEFT ", left_mask),
        ("RIGHT", right_mask),
    ]:

        actual = y_true[mask]
        prediction = y_pred[mask]

        if len(actual) == 0:
            continue

        correct_direction = (
            actual * prediction > 0
        )

        accuracy = (
            np.mean(correct_direction) * 100
        )

        mae = np.mean(
            np.abs(actual - prediction)
        )

        print(
            f"{name}: "
            f"samples={len(actual):7,}  "
            f"direction={accuracy:6.2f}%  "
            f"MAE={mae:.6f}"
        )

    # ============================================================
    # DIRECTION BREAKDOWN
    # ============================================================

    print("\n===== DIRECTION BREAKDOWN =====")

    mask = np.abs(y_true) >= 0.05

    actual = y_true[mask]
    prediction = y_pred[mask]

    actual_left = actual < 0
    actual_right = actual > 0

    pred_left = prediction < 0
    pred_right = prediction > 0

    true_left_pred_left = np.sum(
        actual_left & pred_left
    )

    true_left_pred_right = np.sum(
        actual_left & pred_right
    )

    true_right_pred_right = np.sum(
        actual_right & pred_right
    )

    true_right_pred_left = np.sum(
        actual_right & pred_left
    )

    print(
        f"Actual LEFT  -> Predicted LEFT : "
        f"{true_left_pred_left:,}"
    )

    print(
        f"Actual LEFT  -> Predicted RIGHT: "
        f"{true_left_pred_right:,}"
    )

    print(
        f"Actual RIGHT -> Predicted RIGHT: "
        f"{true_right_pred_right:,}"
    )

    print(
        f"Actual RIGHT -> Predicted LEFT : "
        f"{true_right_pred_left:,}"
    )

    # ============================================================
    # STRONG STEERING
    # ============================================================

    print("\n===== STRONG STEERING (|steer| >= 0.2) =====")

    mask = np.abs(y_true) >= 0.2

    actual = y_true[mask]
    prediction = y_pred[mask]

    if len(actual) > 0:

        mae = np.mean(
            np.abs(actual - prediction)
        )

        direction_accuracy = (
            np.mean(actual * prediction > 0)
            * 100
        )

        print(f"Samples:             {len(actual):,}")
        print(
            f"Actual mean abs:     "
            f"{np.mean(np.abs(actual)):.6f}"
        )
        print(
            f"Prediction mean abs: "
            f"{np.mean(np.abs(prediction)):.6f}"
        )
        print(f"MAE:                 {mae:.6f}")
        print(
            f"Direction accuracy:  "
            f"{direction_accuracy:.2f}%"
        )

    # ============================================================
    # EXAMPLES
    # ============================================================

    print("\n===== EXAMPLES OF STRONG STEERING =====")

    mask = np.abs(y_true) >= 0.2

    actual = y_true[mask]
    prediction = y_pred[mask]

    if len(actual) > 0:

        order = np.argsort(
            -np.abs(actual)
        )

        print(
            "\n   Actual       Prediction       Error"
        )

        for idx in order[:30]:

            error = (
                prediction[idx] - actual[idx]
            )

            print(
                f"{actual[idx]:10.4f}"
                f"{prediction[idx]:15.4f}"
                f"{error:15.4f}"
            )

    # ============================================================
    # PLOT
    # ============================================================

    mask = np.abs(y_true) >= 0.05

    actual = y_true[mask]
    prediction = y_pred[mask]

    plt.figure(figsize=(10, 8))

    plt.scatter(
        actual,
        prediction,
        s=1,
        alpha=0.15
    )

    min_value = min(
        np.min(actual),
        np.min(prediction)
    )

    max_value = max(
        np.max(actual),
        np.max(prediction)
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Actual Steering")
    plt.ylabel("Predicted Steering")

    plt.title(
        "Actual vs Predicted Steering "
        "(|steer| >= 0.05)"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_steering_timeseries(y_true, y_pred, num_samples=2000):
    """
    Plot actual and predicted steering over a consecutive sequence
    of test samples.
    """

    num_samples = min(num_samples, len(y_true))

    actual = y_true[:num_samples]
    predicted = y_pred[:num_samples]

    plt.figure(figsize=(16, 6))

    plt.plot(actual, label="Actual Steering")
    plt.plot(predicted, label="Predicted Steering")

    plt.xlabel("Sample")
    plt.ylabel("Steering")
    plt.title(f"Actual vs Predicted Steering - First {num_samples} Samples")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def analyze_steering_lag(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    max_lag: int = 50,
) -> None:
    """
    Analyze the temporal alignment between actual and predicted steering.

    Tests different time/sample offsets and calculates the correlation
    between actual steering and shifted predictions.

    Parameters
    ----------
    y_actual : np.ndarray
        Actual steering values.

    y_predicted : np.ndarray
        Model-predicted steering values.

    max_lag : int
        Maximum number of samples to test in both directions.
    """

    y_actual = np.asarray(y_actual).flatten()
    y_predicted = np.asarray(y_predicted).flatten()

    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual and predicted arrays must have the same length.")

    lags = range(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:

        if lag < 0:
            actual = y_actual[:lag]
            predicted = y_predicted[-lag:]

        elif lag > 0:
            actual = y_actual[lag:]
            predicted = y_predicted[:-lag]

        else:
            actual = y_actual
            predicted = y_predicted

        correlation = np.corrcoef(actual, predicted)[0, 1]
        correlations.append(correlation)

    best_index = int(np.argmax(correlations))
    best_lag = list(lags)[best_index]
    best_correlation = correlations[best_index]

    print("\n" + "=" * 60)
    print("STEERING TEMPORAL / LAG ANALYSIS")
    print("=" * 60)

    print(f"Best lag:         {best_lag:+d} samples")
    print(f"Best correlation: {best_correlation:.6f}")

    zero_index = list(lags).index(0)

    print(f"Correlation lag 0: {correlations[zero_index]:.6f}")

    if best_lag == 0:
        print("Model prediction is best aligned with actual steering.")
    elif best_lag > 0:
        print(
            "Prediction is temporally ahead/behind actual steering "
            f"by approximately {abs(best_lag)} samples."
        )
    else:
        print(
            "Prediction is temporally shifted relative to actual steering "
            f"by approximately {abs(best_lag)} samples."
        )

    plt.figure(figsize=(12, 5))
    plt.plot(list(lags), correlations)
    plt.axvline(0, linestyle="--")
    plt.axvline(best_lag, linestyle="--")

    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.title("Steering Prediction - Temporal Alignment")
    plt.grid(True)

    plt.show()



def show_worst_steering_predictions(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    n: int = 100,
) -> None:
    """
    Display the samples with the largest absolute steering prediction errors.
    """

    y_actual = np.asarray(y_actual).flatten()
    y_predicted = np.asarray(y_predicted).flatten()

    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual and predicted arrays must have the same length.")

    errors = np.abs(y_actual - y_predicted)

    # Indices of the n largest errors
    worst_indices = np.argsort(errors)[-n:][::-1]

    print("\n" + "=" * 65)
    print(f"WORST {n} STEERING PREDICTIONS")
    print("=" * 65)

    print(
        f"{'Index':>8} "
        f"{'Actual':>12} "
        f"{'Prediction':>12} "
        f"{'Error':>12}"
    )

    print("-" * 65)

    for index in worst_indices:
        actual = y_actual[index]
        prediction = y_predicted[index]
        error = errors[index]

        print(
            f"{index:8d} "
            f"{actual:12.4f} "
            f"{prediction:12.4f} "
            f"{error:12.4f}"
        )

def plot_prediction_context(
    y_true,
    y_pred,
    indices,
    context=50
):
    import matplotlib.pyplot as plt
    import numpy as np

    for idx in indices:
        start = max(0, idx - context)
        end = min(len(y_true), idx + context + 1)

        x = np.arange(start, end)

        plt.figure(figsize=(14, 5))

        plt.plot(
            x,
            y_true[start:end],
            label="Actual Steering"
        )

        plt.plot(
            x,
            y_pred[start:end],
            label="Predicted Steering"
        )

        plt.axvline(
            idx,
            linestyle="--",
            label=f"Worst sample {idx}"
        )

        plt.title(
            f"Prediction Context around sample {idx}"
        )

        plt.xlabel("Sample")
        plt.ylabel("Steering")
        plt.legend()
        plt.grid(True)
        plt.show()

# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load and prepare the CARLA telemetry dataset."""

    print("Loading dataset...")

    dataset = pd.read_csv(
        GLOBAL_CONFIG.pipeline.csv_path
    )

    dataset["timestamp"] = pd.to_datetime(
        dataset["timestamp"],
        unit="s",
    )

    print(
        f"Loaded {len(dataset):,} samples."
    )

    return dataset


# ============================================================
# BASIC DATASET ANALYSIS
# ============================================================

def print_dataset_overview(
    dataset: pd.DataFrame,
) -> None:
    """Print basic information about the dataset."""

    print("\n===== DATASET OVERVIEW =====")

    print(
        f"Samples:       {len(dataset):,}"
    )

    print(
        f"Vehicles:      {dataset['vehicle_id'].nunique():,}"
    )

    print(
        f"Columns:       {len(dataset.columns):,}"
    )

    print(
        f"Start:         {dataset['timestamp'].min()}"
    )

    print(
        f"End:           {dataset['timestamp'].max()}"
    )

    print("\n===== COLUMNS =====")

    print(
        dataset.columns.tolist()
    )

    print("\n===== MISSING VALUES =====")

    print(
        dataset.isnull()
        .sum()
        .to_string()
    )


# ============================================================
# VEHICLE ANALYSIS
# ============================================================

def print_vehicle_statistics(
    dataset: pd.DataFrame,
) -> None:
    """Print number of samples and recording duration per vehicle."""

    print("\n===== VEHICLE COUNT =====")

    print(
        dataset["vehicle_id"].nunique()
    )

    print("\n===== SAMPLES PER VEHICLE =====")

    vehicle_counts = (
        dataset["vehicle_id"]
        .value_counts()
        .sort_index()
    )

    print(
        vehicle_counts.to_string()
    )

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
        (
            vehicle_stats["end"]
            - vehicle_stats["start"]
        )
        .dt.total_seconds()
        / 60.0
    )

    print(
        vehicle_stats.to_string()
    )


# ============================================================
# CAPTURE GROUPS
# ============================================================

def assign_capture_group(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign capture group based on the four CARLA recordings.

    Group 1: vehicle IDs 197-235
    Group 2: vehicle IDs 348-386
    Group 3: vehicle IDs 473-511
    Group 4: vehicle IDs 636-674
    """

    dataset = dataset.copy()

    dataset["capture_group"] = pd.NA

    dataset.loc[
        dataset["vehicle_id"].between(197, 235),
        "capture_group",
    ] = 1

    dataset.loc[
        dataset["vehicle_id"].between(348, 386),
        "capture_group",
    ] = 2

    dataset.loc[
        dataset["vehicle_id"].between(473, 511),
        "capture_group",
    ] = 3

    dataset.loc[
        dataset["vehicle_id"].between(636, 674),
        "capture_group",
    ] = 4

    if dataset["capture_group"].isna().any():

        unknown_ids = (
            dataset.loc[
                dataset["capture_group"].isna(),
                "vehicle_id",
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown vehicle IDs found: {unknown_ids}"
        )

    dataset["capture_group"] = (
        dataset["capture_group"]
        .astype(int)
    )

    return dataset


def print_capture_groups(
    dataset: pd.DataFrame,
) -> None:
    """Print statistics for each CARLA capture group."""

    print("\n===== VEHICLE GROUPS =====")

    groups = (
        dataset
        .groupby("capture_group")
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            samples=("timestamp", "size"),
            vehicles=("vehicle_id", "nunique"),
        )
    )

    for group_id, row in groups.iterrows():

        vehicles = (
            dataset.loc[
                dataset["capture_group"] == group_id,
                "vehicle_id",
            ]
            .unique()
        )

        print(
            f"Group {group_id}: "
            f"{vehicles.min()} - {vehicles.max()}"
        )

        print(
            f"  Start:   {row['start']}"
        )

        print(
            f"  End:     {row['end']}"
        )

        print(
            f"  Samples: {row['samples']:,}"
        )

        print(
            f"  Vehicles: {row['vehicles']}"
        )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_split(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create group-based train/validation/test split.

    Train:
        Groups 1 and 2

    Validation:
        Group 3

    Test:
        Group 4
    """

    train = dataset[
        dataset["capture_group"].isin([1, 2])
    ].copy()

    validation = dataset[
        dataset["capture_group"] == 3
    ].copy()

    test = dataset[
        dataset["capture_group"] == 4
    ].copy()

    return train, validation, test


# ============================================================
# SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print statistics for train, validation and test sets."""

    print("\n===== SPLIT STATISTICS =====")

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        print(f"\n--- {name} ---")

        print(
            f"Samples: {len(data):,}"
        )

        print(
            "Groups:  "
            f"{sorted(data['capture_group'].unique())}"
        )

        print(
            f"Vehicles: "
            f"{data['vehicle_id'].nunique()}"
        )

        print(
            f"Steer mean: "
            f"{data[TARGET].mean():.6f}"
        )

        print(
            f"Steer std:  "
            f"{data[TARGET].std():.6f}"
        )

        baseline_mae = (
            data[TARGET]
            .abs()
            .mean()
        )

        print(
            "Steer MAE baseline "
            "(predict 0): "
            f"{baseline_mae:.6f}"
        )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def print_feature_statistics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print feature statistics for all dataset splits."""

    print("\n===== FEATURE STATISTICS =====")

    for feature in FEATURES:

        print(f"\n--- {feature} ---")

        for name, data in [
            ("TRAIN", train),
            ("VAL", validation),
            ("TEST", test),
        ]:

            values = data[feature]

            print(
                f"{name:>5}: "
                f"mean={values.mean():.6f}, "
                f"std={values.std():.6f}, "
                f"min={values.min():.6f}, "
                f"max={values.max():.6f}"
            )


# ============================================================
# FEATURE QUANTILES
# ============================================================

def print_feature_quantiles(
    dataset: pd.DataFrame,
) -> None:
    """Print feature quantiles."""

    print("\n===== FEATURE QUANTILES =====")

    quantiles = [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
    ]

    for feature in FEATURES:

        print(f"\n--- {feature} ---")

        print(
            dataset[feature]
            .quantile(quantiles)
            .to_string()
        )


# ============================================================
# STEERING DISTRIBUTION
# ============================================================

def print_steering_statistics(
    dataset: pd.DataFrame,
) -> None:
    """Analyze steering distribution."""

    print("\n===== STEERING DISTRIBUTION =====")

    print(
        dataset[TARGET]
        .describe()
    )

    print("\n===== STEERING QUANTILES =====")

    print(
        dataset[TARGET]
        .quantile(
            [
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )


# ============================================================
# STEERING MAGNITUDE ANALYSIS
# ============================================================

def analyze_steering_by_bins(
    dataset: pd.DataFrame,
) -> None:
    """
    Analyze how much of the dataset is occupied by
    different steering magnitudes.
    """

    print(
        "\n===== STEERING ERROR BASELINE BY MAGNITUDE ====="
    )

    abs_steer = dataset[TARGET].abs()

    bins = [
        0.0,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.4,
        np.inf,
    ]

    labels = [
        "0-0.005",
        "0.005-0.01",
        "0.01-0.025",
        "0.025-0.05",
        "0.05-0.1",
        "0.1-0.2",
        "0.2-0.4",
        ">0.4",
    ]

    steer_bins = pd.cut(
        abs_steer,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        dataset
        .assign(steer_bin=steer_bins)
        .groupby(
            "steer_bin",
            observed=True,
        )
        .agg(
            samples=("steer", "size"),
            steer_mean=("steer", "mean"),
            steer_std=("steer", "std"),
            steer_abs_mean=(
                "steer",
                lambda x: x.abs().mean(),
            ),
        )
    )

    result["percentage"] = (
        result["samples"]
        / len(dataset)
        * 100
    )

    print(
        result.to_string()
    )


# ============================================================
# GROUP STEERING STATISTICS
# ============================================================

def print_group_steering_statistics(
    dataset: pd.DataFrame,
) -> None:
    """Print steering and vehicle-control statistics per group."""

    print("\n===== GROUP STEERING STATISTICS =====")

    result = (
        dataset
        .groupby("capture_group")
        .agg(
            samples=("steer", "size"),
            steer_mean=("steer", "mean"),
            steer_std=("steer", "std"),
            steer_min=("steer", "min"),
            steer_max=("steer", "max"),
            lane_offset_mean=(
                "lane_offset",
                "mean",
            ),
            lane_offset_std=(
                "lane_offset",
                "std",
            ),
            heading_error_mean=(
                "heading_error",
                "mean",
            ),
            heading_error_std=(
                "heading_error",
                "std",
            ),
        )
    )

    print(
        result.to_string()
    )


# ============================================================
# STEERING DISTRIBUTION HISTOGRAM
# ============================================================

def plot_steering_histogram(
    dataset: pd.DataFrame,
) -> None:
    """Plot steering distribution for all capture groups."""

    plt.figure(
        figsize=(12, 7)
    )

    for group_id in sorted(
        dataset["capture_group"].unique()
    ):

        values = dataset.loc[
            dataset["capture_group"] == group_id,
            TARGET,
        ]

        plt.hist(
            values,
            bins=150,
            density=True,
            alpha=0.45,
            label=f"Group {group_id}",
        )

    plt.xlabel("Steering")
    plt.ylabel("Density")
    plt.title(
        "Steering Distribution by Group"
    )

    plt.grid(
        True,
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# STEERING HISTOGRAM - ZOOMED
# ============================================================

def plot_steering_histogram_zoomed(
    dataset: pd.DataFrame,
) -> None:
    """Plot zoomed steering distribution."""

    plt.figure(
        figsize=(12, 7)
    )

    for group_id in sorted(
        dataset["capture_group"].unique()
    ):

        values = dataset.loc[
            dataset["capture_group"] == group_id,
            TARGET,
        ]

        plt.hist(
            values,
            bins=150,
            density=True,
            alpha=0.45,
            label=f"Group {group_id}",
        )

    plt.xlim(
        -0.15,
        0.15,
    )

    plt.xlabel("Steering")
    plt.ylabel("Density")

    plt.title(
        "Steering Distribution by Group - Zoomed"
    )

    plt.grid(
        True,
        alpha=0.2,
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_steering_relationships(
    dataset: pd.DataFrame,
) -> None:
    """Plot relationships between steering and important features."""

    # --------------------------------------------------------
    # Lane offset
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        dataset["lane_offset"],
        dataset[TARGET],
        s=1,
        alpha=0.1,
    )

    plt.xlabel("Lane Offset")
    plt.ylabel("Steering")

    plt.title(
        "Lane Offset vs Steering"
    )

    plt.grid(
        True,
        alpha=0.2,
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Heading error
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        dataset["heading_error"],
        dataset[TARGET],
        s=1,
        alpha=0.1,
    )

    plt.xlabel("Heading Error")
    plt.ylabel("Steering")

    plt.title(
        "Heading Error vs Steering"
    )

    plt.grid(
        True,
        alpha=0.2,
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CORRELATION
# ============================================================

def print_correlations(
    dataset: pd.DataFrame,
) -> None:
    """Print Pearson correlation matrix."""

    columns = FEATURES + [TARGET]

    print(
        "\n===== CORRELATION MATRIX ====="
    )

    print(
        dataset[columns]
        .corr()
        .round(4)
        .to_string()
    )


# ============================================================
# SCALING
# ============================================================

def scale_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Scale features using StandardScaler.

    IMPORTANT:
    The scaler is fitted ONLY on training data.
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_validation = scaler.transform(
        validation[FEATURES]
    )

    X_test = scaler.transform(
        test[FEATURES]
    )

    y_train = train[TARGET].to_numpy(
        dtype=np.float32
    )

    y_validation = validation[TARGET].to_numpy(
        dtype=np.float32
    )

    y_test = test[TARGET].to_numpy(
        dtype=np.float32
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        scaler,
    )


# ============================================================
# SCALER STATISTICS
# ============================================================

def print_scaler_statistics(
    scaler: StandardScaler,
) -> None:
    """Print fitted StandardScaler parameters."""

    print("\n===== Standard SCALER =====")

    for feature, mean, scale in zip(
        FEATURES,
        scaler.mean_,
        scaler.scale_,
    ):

        print(
            f"{feature:20s} "
            f"mean={mean:.8f} "
            f"scale={scale:.8f}"
        )


# ============================================================
# SCALED DATA STATISTICS
# ============================================================

def print_scaled_data_statistics(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
) -> None:
    """Print statistics of scaled datasets."""

    print("\n===== SCALED DATA =====")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_validation.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    train_scaled = pd.DataFrame(
        X_train,
        columns=FEATURES,
    )

    print(
        "\n===== TRAIN SCALED STATISTICS ====="
    )

    print(
        train_scaled
        .describe()
        .loc[
            ["mean", "std", "min", "max"]
        ]
    )

def inspect_prediction_context(
    X_test,
    y_test,
    y_pred,
    sample_idx: int,
    context: int = 10
):
    y_pred = y_pred.reshape(-1)

    start = max(0, sample_idx - context)
    end = min(len(y_test), sample_idx + context + 1)

    print("=" * 70)
    print(f"PREDICTION CONTEXT AROUND SAMPLE {sample_idx}")
    print("=" * 70)

    for i in range(start, end):
        marker = " <-- WORST" if i == sample_idx else ""

        print(
            f"{i:8d} | "
            f"actual={y_test[i]: .6f} | "
            f"pred={y_pred[i]: .6f} | "
            f"error={abs(y_test[i] - y_pred[i]): .6f}"
            f"{marker}"
        )


def inspect_input_context(X_test, y_test, y_pred, sample_idx, context=3):
    y_pred = y_pred.reshape(-1)

    start = max(0, sample_idx - context)
    end = min(len(X_test), sample_idx + context + 1)

    print("=" * 80)
    print(f"INPUT CONTEXT AROUND SAMPLE {sample_idx}")
    print("=" * 80)

    for i in range(start, end):
        print(f"\nSample {i}")
        print(f"  Actual:     {y_test[i]: .6f}")
        print(f"  Prediction: {y_pred[i]: .6f}")
        print(f"  Error:      {abs(y_test[i] - y_pred[i]): .6f}")
        print(f"  X:          {X_test[i]}")


def inspect_nearest_training_samples(
    X_train,
    y_train,
    X_test,
    y_test,
    sample_idx: int,
    n_neighbors: int = 20,
) -> None:
    """
    Finds the nearest training samples to a selected test sample.

    The comparison is performed in the scaled feature space.

    This is useful for determining whether a large prediction error
    occurs in an input region that was well represented during training.
    """

    feature_names = [
        "speed",
        "acceleration",
        "distance",
        "lane_offset",
        "heading_error",
    ]

    # ------------------------------------------------------------------
    # Select test sample
    # ------------------------------------------------------------------

    query = X_test[sample_idx]
    actual = y_test[sample_idx]

    # Make sure the target is a scalar
    actual = float(np.asarray(actual).squeeze())

    # ------------------------------------------------------------------
    # Calculate Euclidean distance to every training sample
    # ------------------------------------------------------------------

    distances = np.linalg.norm(X_train - query, axis=1)

    # Get indices of closest samples
    nearest_indices = np.argsort(distances)[:n_neighbors]

    # ------------------------------------------------------------------
    # Print query sample
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(f"NEAREST TRAINING SAMPLES FOR TEST SAMPLE {sample_idx}")
    print("=" * 80)

    print()
    print("TEST SAMPLE")
    print("-" * 80)

    print(f"Actual steer: {actual:.6f}")

    print("Scaled X:")
    for i, name in enumerate(feature_names):
        print(f"  {name:<15}: {query[i]: .6f}")

    # ------------------------------------------------------------------
    # Print nearest neighbours
    # ------------------------------------------------------------------

    print()
    print("NEAREST TRAINING SAMPLES")
    print("-" * 80)

    print(
        f"{'Rank':>4} "
        f"{'Train index':>12} "
        f"{'Distance':>12} "
        f"{'Train steer':>14}"
    )

    print("-" * 80)

    for rank, train_idx in enumerate(nearest_indices, start=1):

        distance = distances[train_idx]
        train_steer = float(np.asarray(y_train[train_idx]).squeeze())

        print(
            f"{rank:4d} "
            f"{train_idx:12d} "
            f"{distance:12.6f} "
            f"{train_steer:14.6f}"
        )

    # ------------------------------------------------------------------
    # Summary statistics of neighbours
    # ------------------------------------------------------------------

    neighbour_targets = np.asarray(y_train)[nearest_indices].reshape(-1)

    print()
    print("NEIGHBOUR TARGET STATISTICS")
    print("-" * 80)

    print(f"Mean steer:     {np.mean(neighbour_targets): .6f}")
    print(f"Std steer:      {np.std(neighbour_targets): .6f}")
    print(f"Min steer:      {np.min(neighbour_targets): .6f}")
    print(f"Max steer:      {np.max(neighbour_targets): .6f}")
    print(f"Mean abs steer: {np.mean(np.abs(neighbour_targets)): .6f}")

    # ------------------------------------------------------------------
    # Compare query against nearest neighbour
    # ------------------------------------------------------------------

    closest_idx = nearest_indices[0]

    print()
    print("CLOSEST TRAINING SAMPLE")
    print("-" * 80)

    print(f"Training index: {closest_idx}")
    print(f"Distance:       {distances[closest_idx]:.6f}")
    print(f"Training steer: {float(y_train[closest_idx]):.6f}")
    print(f"Test steer:     {actual:.6f}")

    print()
    print("Feature differences (test - closest train):")
    print("-" * 80)

    for i, name in enumerate(feature_names):
        difference = query[i] - X_train[closest_idx, i]

        print(
            f"  {name:<15}: "
            f"{difference: .6f}"
        )

    print("=" * 80)

# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    dataset = load_dataset()
    dataset = (
        dataset
        .sort_values(
            by=["vehicle_id", "timestamp"]
        )
        .reset_index(drop=True)
    )
    # --------------------------------------------------------
    # 2. Basic dataset analysis
    # --------------------------------------------------------

    #print_dataset_overview(
    #    dataset
    #)

    #print_vehicle_statistics(
    #    dataset
    #)

    # --------------------------------------------------------
    # 3. Capture groups
    # --------------------------------------------------------

    #dataset = assign_capture_group(
    #    dataset
    #)

    #print_capture_groups(
    #    dataset
    #)

    # --------------------------------------------------------
    # 4. Steering analysis
    # --------------------------------------------------------

    #print_steering_statistics(
    #    dataset
    #)

    #analyze_steering_by_bins(
    #    dataset
    #)

    #print_group_steering_statistics(
    #    dataset
    #)

    # --------------------------------------------------------
    # 5. Split dataset
    # --------------------------------------------------------

    #train, validation, test = create_split(
    #    dataset
    #)

    #print_split_statistics(
    #    train,
    #    validation,
    #    test,
    #)

    # --------------------------------------------------------
    # 6. Feature statistics
    # --------------------------------------------------------

    #print_feature_statistics(
    #    train,
    #    validation,
    #    test,
    #)

    # Quantiles should preferably be inspected on TRAIN
    # because TRAIN is what the model actually learns from.

    #print_feature_quantiles(
    #    train
    #)

    # --------------------------------------------------------
    # 7. Correlations
    # --------------------------------------------------------

    #print_correlations(
    #    train
    #)

    # --------------------------------------------------------
    # 8. Scaling
    # --------------------------------------------------------

    #(
    #    X_train,
    #    X_validation,
    #    X_test,
    #    y_train,
    #    y_validation,
    #    y_test,
    #    scaler,
    #) = scale_data(
    #    train,
    #    validation,
    #    test,
    #)

    print_scaled_data_statistics(
        X_train,
        X_validation,
        X_test,
    )

    print_scaler_statistics(
        scaler
    )

    # --------------------------------------------------------
    # 9. Plots
    # --------------------------------------------------------

    plot_steering_histogram(
        dataset
    )

    plot_steering_histogram_zoomed(
        dataset
    )

    plot_steering_relationships(
        dataset
    )

    return (
        train,
        validation,
        test,
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        scaler,
    )

def inspect_original_input_context(
    X_scaled,
    y_test,
    y_pred,
    scaler,
    sample_idx,
    context=5
):
    """
    Displays scaled and original feature values around a selected sample.
    Used to verify that preprocessing/scaling is working correctly.
    """

    X_original = scaler.inverse_transform(X_scaled)

    feature_names = [
        "speed",
        "acceleration",
        "distance",
        "lane_offset",
        "heading_error",
    ]

    start = max(0, sample_idx - context)
    end = min(len(X_scaled), sample_idx + context + 1)

    print("=" * 80)
    print(f"ORIGINAL INPUT CONTEXT AROUND SAMPLE {sample_idx}")
    print("=" * 80)

    for i in range(start, end):

        print(f"\nSample {i}")

        print(f"  Actual:      {y_test[i]: .6f}")
        print(f"  Prediction:  {y_pred[i, 0]: .6f}")
        print(f"  Error:       {abs(y_test[i] - y_pred[i, 0]): .6f}")

        print("  Scaled X:")
        for name, value in zip(feature_names, X_scaled[i]):
            print(f"    {name:15s}: {value: .6f}")

        print("  Original X:")
        for name, value in zip(feature_names, X_original[i]):
            print(f"    {name:15s}: {value: .6f}")


def analyze_knn_context(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    sample_idx: int,
    k_values=(1, 5, 10, 20, 50),
):
    """
    Analyze the local training-data neighborhood of a test sample.

    For several K values, reports:
      - mean/std/min/max steering of nearest training samples
      - KNN steering estimate
      - distance to the nearest training sample
      - neural-network prediction
      - actual steering value
    """

    # Flatten prediction array from (N, 1) -> (N,)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    x = X_test[sample_idx].reshape(1, -1)

    max_k = max(k_values)

    # Find nearest training samples
    nn = NearestNeighbors(
        n_neighbors=max_k,
        metric="euclidean"
    )

    nn.fit(X_train)

    distances, indices = nn.kneighbors(x)

    distances = distances[0]
    indices = indices[0]

    print("=" * 80)
    print(f"KNN LOCAL ANALYSIS - TEST SAMPLE {sample_idx}")
    print("=" * 80)

    print(f"\nTest steering:       {y_test[sample_idx]: .6f}")
    print(f"NN prediction:       {y_pred[sample_idx]: .6f}")
    print(f"NN absolute error:   {abs(y_test[sample_idx] - y_pred[sample_idx]): .6f}")

    print("\nTest input:")
    print(X_test[sample_idx])

    print("\n" + "-" * 80)
    print("KNN STEERING ANALYSIS")
    print("-" * 80)

    for k in k_values:

        k_distances = distances[:k]
        k_indices = indices[:k]
        k_steering = y_train[k_indices]

        knn_prediction = np.mean(k_steering)

        print(f"\nK = {k}")
        print(f"  Distance range:    {k_distances[0]:.6f} -> {k_distances[-1]:.6f}")
        print(f"  KNN prediction:    {knn_prediction: .6f}")
        print(f"  Steering mean:     {np.mean(k_steering): .6f}")
        print(f"  Steering std:      {np.std(k_steering): .6f}")
        print(f"  Steering min:      {np.min(k_steering): .6f}")
        print(f"  Steering max:      {np.max(k_steering): .6f}")
        print(
            f"  KNN error:         "
            f"{abs(y_test[sample_idx] - knn_prediction): .6f}"
        )

    print("\n" + "-" * 80)
    print(f"TOP {max_k} NEAREST TRAINING SAMPLES")
    print("-" * 80)

    print(
        f"{'Rank':>5} "
        f"{'Train index':>12} "
        f"{'Distance':>12} "
        f"{'Train steer':>14}"
    )

    for rank, (distance, train_idx) in enumerate(
        zip(distances, indices),
        start=1
    ):
        print(
            f"{rank:5d} "
            f"{train_idx:12d} "
            f"{distance:12.6f} "
            f"{y_train[train_idx]:14.6f}"
        )

    print("=" * 80)

def analyze_near_zero_predictions(
    y_true,
    y_pred,
    thresholds=(0.001, 0.005, 0.01, 0.025, 0.05),
):
    """
    Analyze model behaviour for samples where actual steering
    is close to zero.

    Compares model predictions against a zero-steering baseline.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    abs_true = np.abs(y_true)

    boundaries = [0.0] + list(thresholds)

    print("=" * 90)
    print("NEAR-ZERO STEERING ANALYSIS")
    print("=" * 90)

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):

        mask = (abs_true >= lower) & (abs_true < upper)

        if not np.any(mask):
            continue

        actual = y_true[mask]
        pred = y_pred[mask]

        errors = np.abs(actual - pred)

        baseline_mae = np.mean(np.abs(actual))
        model_mae = np.mean(errors)

        positive_actual = np.sum(actual > 0)
        negative_actual = np.sum(actual < 0)
        zero_actual = np.sum(actual == 0)

        positive_pred = np.sum(pred > 0)
        negative_pred = np.sum(pred < 0)
        near_zero_pred = np.sum(np.abs(pred) < 0.005)

        print()
        print("-" * 90)
        print(
            f"|steer| [{lower:.3f}, {upper:.3f})"
        )
        print("-" * 90)

        print(f"Samples:          {len(actual)}")
        print(f"Percentage:       {100 * len(actual) / len(y_true):.2f}%")

        print()
        print("ACTUAL")
        print(f"  mean:           {np.mean(actual): .6f}")
        print(f"  std:            {np.std(actual): .6f}")
        print(f"  min:            {np.min(actual): .6f}")
        print(f"  max:            {np.max(actual): .6f}")

        print()
        print("PREDICTION")
        print(f"  mean:           {np.mean(pred): .6f}")
        print(f"  std:            {np.std(pred): .6f}")
        print(f"  min:            {np.min(pred): .6f}")
        print(f"  max:            {np.max(pred): .6f}")

        print()
        print("ERROR")
        print(f"  model MAE:      {model_mae:.6f}")
        print(f"  baseline MAE:   {baseline_mae:.6f}")

        if baseline_mae > 0:
            improvement = (
                (baseline_mae - model_mae)
                / baseline_mae
                * 100
            )
            print(f"  improvement:    {improvement: .2f}%")

        print()
        print("ACTUAL DIRECTION")
        print(f"  positive:       {positive_actual}")
        print(f"  negative:       {negative_actual}")
        print(f"  zero:           {zero_actual}")

        print()
        print("PREDICTED DIRECTION")
        print(f"  positive:       {positive_pred}")
        print(f"  negative:       {negative_pred}")
        print(f"  |pred| < 0.005: {near_zero_pred}")

        # How many predictions are surprisingly far from zero?
        for prediction_threshold in (0.01, 0.025, 0.05, 0.1):

            count = np.sum(np.abs(pred) >= prediction_threshold)

            print(
                f"  |pred| >= {prediction_threshold:.3f}: "
                f"{count} "
                f"({100 * count / len(pred):.2f}%)"
            )

    print()
    print("=" * 90)

def analyze_near_zero_features(
    X,
    y,
    feature_names,
    zero_threshold=0.001,
):
    """
    Compare input feature distributions between:
      - near-zero steering samples
      - non-zero steering samples

    Helps determine whether the input features contain enough
    information to distinguish straight driving from steering.
    """

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    zero_mask = np.abs(y) < zero_threshold
    nonzero_mask = ~zero_mask

    X_zero = X[zero_mask]
    X_nonzero = X[nonzero_mask]

    print("=" * 90)
    print("FEATURE ANALYSIS: NEAR-ZERO VS NON-ZERO STEERING")
    print("=" * 90)

    print(f"Near-zero samples:     {len(X_zero)}")
    print(f"Non-zero samples:      {len(X_nonzero)}")
    print(
        f"Near-zero percentage:  "
        f"{100 * len(X_zero) / len(X):.2f}%"
    )

    print()
    print("-" * 90)
    print(
        f"{'Feature':<20}"
        f"{'Zero mean':>15}"
        f"{'Zero std':>15}"
        f"{'Nonzero mean':>15}"
        f"{'Nonzero std':>15}"
    )
    print("-" * 90)

    for i, feature in enumerate(feature_names):

        zero_mean = np.mean(X_zero[:, i])
        zero_std = np.std(X_zero[:, i])

        nonzero_mean = np.mean(X_nonzero[:, i])
        nonzero_std = np.std(X_nonzero[:, i])

        print(
            f"{feature:<20}"
            f"{zero_mean:>15.6f}"
            f"{zero_std:>15.6f}"
            f"{nonzero_mean:>15.6f}"
            f"{nonzero_std:>15.6f}"
        )

    print()
    print("=" * 90)
    print("FEATURE RANGE")
    print("=" * 90)

    for i, feature in enumerate(feature_names):

        print()
        print(feature)

        print(
            f"  Near-zero: "
            f"min={np.min(X_zero[:, i]):.6f}, "
            f"max={np.max(X_zero[:, i]):.6f}"
        )

        print(
            f"  Non-zero:  "
            f"min={np.min(X_nonzero[:, i]):.6f}, "
            f"max={np.max(X_nonzero[:, i]):.6f}"
        )

    print()
    print("=" * 90)

def analyze_feature_steering_relationship(
    X,
    y,
    feature_names,
):
    """
    Analyze the relationship between each input feature and steering.

    Calculates Pearson and Spearman correlations and creates scatter
    plots for the input features against the steering target.
    """

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    print("=" * 90)
    print("FEATURE / STEERING CORRELATION ANALYSIS")
    print("=" * 90)

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    from scipy.stats import pearsonr, spearmanr

    print()
    print(
        f"{'Feature':<20}"
        f"{'Pearson':>15}"
        f"{'Spearman':>15}"
    )
    print("-" * 50)

    correlations = {}

    for i, feature in enumerate(feature_names):

        feature_values = X[:, i]

        pearson_corr, _ = pearsonr(feature_values, y)
        spearman_corr, _ = spearmanr(feature_values, y)

        correlations[feature] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

        print(
            f"{feature:<20}"
            f"{pearson_corr:>15.6f}"
            f"{spearman_corr:>15.6f}"
        )

    # ---------------------------------------------------------
    # Scatter plots
    # ---------------------------------------------------------

    for i, feature in enumerate(feature_names):

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X[:, i],
            y,
            s=2,
            alpha=0.15,
        )

        plt.xlabel(feature)
        plt.ylabel("Steering")
        plt.title(f"{feature} vs Steering")

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    return correlations

import numpy as np


def analyze_near_zero_separability(
    X,
    y,
    feature_names,
    threshold=0.001,
):
    """
    Analyze whether near-zero and non-zero steering samples
    are distinguishable from the input features.
    """

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    near_zero = np.abs(y) < threshold
    non_zero = ~near_zero

    X_zero = X[near_zero]
    X_nonzero = X[non_zero]

    print("=" * 100)
    print("NEAR-ZERO vs NON-ZERO FEATURE SEPARABILITY")
    print("=" * 100)

    print(f"Threshold:             |steering| < {threshold}")
    print(f"Near-zero samples:     {len(X_zero)}")
    print(f"Non-zero samples:      {len(X_nonzero)}")
    print()

    print("-" * 100)
    print(
        f"{'Feature':<20}"
        f"{'Zero mean':>15}"
        f"{'Zero std':>15}"
        f"{'Nonzero mean':>15}"
        f"{'Nonzero std':>15}"
        f"{'Mean diff':>15}"
    )
    print("-" * 100)

    for i, feature in enumerate(feature_names):

        zero_values = X_zero[:, i]
        nonzero_values = X_nonzero[:, i]

        zero_mean = np.mean(zero_values)
        zero_std = np.std(zero_values)

        nonzero_mean = np.mean(nonzero_values)
        nonzero_std = np.std(nonzero_values)

        mean_diff = nonzero_mean - zero_mean

        print(
            f"{feature:<20}"
            f"{zero_mean:>15.6f}"
            f"{zero_std:>15.6f}"
            f"{nonzero_mean:>15.6f}"
            f"{nonzero_std:>15.6f}"
            f"{mean_diff:>15.6f}"
        )

    # ---------------------------------------------------------
    # Standardized mean difference
    # ---------------------------------------------------------

    print()
    print("-" * 100)
    print("STANDARDIZED MEAN DIFFERENCE")
    print("-" * 100)

    print(
        f"{'Feature':<20}"
        f"{'Effect size':>20}"
    )

    effect_sizes = {}

    for i, feature in enumerate(feature_names):

        zero_values = X_zero[:, i]
        nonzero_values = X_nonzero[:, i]

        mean_zero = np.mean(zero_values)
        mean_nonzero = np.mean(nonzero_values)

        std_zero = np.std(zero_values)
        std_nonzero = np.std(nonzero_values)

        pooled_std = np.sqrt(
            (std_zero ** 2 + std_nonzero ** 2) / 2
        )

        if pooled_std > 0:
            effect_size = (
                (mean_nonzero - mean_zero)
                / pooled_std
            )
        else:
            effect_size = 0.0

        effect_sizes[feature] = effect_size

        print(
            f"{feature:<20}"
            f"{effect_size:>20.6f}"
        )

    # ---------------------------------------------------------
    # Feature ranges
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("FEATURE RANGE OVERLAP")
    print("=" * 100)

    for i, feature in enumerate(feature_names):

        zero_min = np.min(X_zero[:, i])
        zero_max = np.max(X_zero[:, i])

        nonzero_min = np.min(X_nonzero[:, i])
        nonzero_max = np.max(X_nonzero[:, i])

        overlap_min = max(zero_min, nonzero_min)
        overlap_max = min(zero_max, nonzero_max)

        if overlap_min <= overlap_max:
            overlap = "YES"
        else:
            overlap = "NO"

        print(f"\n{feature}")
        print(
            f"  Near-zero range: "
            f"{zero_min:.6f} -> {zero_max:.6f}"
        )
        print(
            f"  Non-zero range:  "
            f"{nonzero_min:.6f} -> {nonzero_max:.6f}"
        )
        print(
            f"  Range overlap:   {overlap}"
        )

    return effect_sizes


def analyze_near_zero_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    threshold=0.001,
    k=20,
    n_samples=1000,
    random_state=42,
):
    """
    Analyze whether near-zero steering test samples have
    near-zero steering neighbours in the training dataset.

    X_train / X_test should already be StandardScaler transformed.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).reshape(-1)

    # ---------------------------------------------------------
    # Select near-zero test samples
    # ---------------------------------------------------------

    near_zero_mask = np.abs(y_test) < threshold
    near_zero_indices = np.where(near_zero_mask)[0]

    print("=" * 100)
    print("KNN LOCAL ANALYSIS - NEAR-ZERO TEST SAMPLES")
    print("=" * 100)

    print(f"Threshold:                 |steering| < {threshold}")
    print(f"Available near-zero:       {len(near_zero_indices)}")

    # Random subset
    rng = np.random.default_rng(random_state)

    if len(near_zero_indices) > n_samples:
        selected_indices = rng.choice(
            near_zero_indices,
            size=n_samples,
            replace=False,
        )
    else:
        selected_indices = near_zero_indices

    print(f"Samples analyzed:          {len(selected_indices)}")
    print(f"K:                         {k}")

    X_query = X_test[selected_indices]
    y_query = y_test[selected_indices]

    # ---------------------------------------------------------
    # Fit KNN on training data
    # ---------------------------------------------------------

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1,
    )

    knn.fit(X_train)

    distances, indices = knn.kneighbors(X_query)

    neighbor_y = y_train[indices]

    # ---------------------------------------------------------
    # Basic neighbour statistics
    # ---------------------------------------------------------

    neighbor_abs_y = np.abs(neighbor_y)

    neighbor_near_zero = neighbor_abs_y < threshold

    near_zero_ratio = np.mean(
        neighbor_near_zero,
        axis=1,
    )

    neighbor_mean = np.mean(
        neighbor_y,
        axis=1,
    )

    neighbor_std = np.std(
        neighbor_y,
        axis=1,
    )

    neighbor_min = np.min(
        neighbor_y,
        axis=1,
    )

    neighbor_max = np.max(
        neighbor_y,
        axis=1,
    )

    knn_prediction = neighbor_mean

    knn_error = np.abs(
        y_query - knn_prediction
    )

    # ---------------------------------------------------------
    # Distance statistics
    # ---------------------------------------------------------

    nearest_distance = distances[:, 0]

    mean_distance = np.mean(
        distances,
        axis=1,
    )

    # ---------------------------------------------------------
    # Print global results
    # ---------------------------------------------------------

    print()
    print("-" * 100)
    print("NEAREST-NEIGHBOUR DISTANCE")
    print("-" * 100)

    print(
        f"Nearest distance mean:     "
        f"{np.mean(nearest_distance):.6f}"
    )

    print(
        f"Nearest distance median:   "
        f"{np.median(nearest_distance):.6f}"
    )

    print(
        f"Nearest distance min:      "
        f"{np.min(nearest_distance):.6f}"
    )

    print(
        f"Nearest distance max:      "
        f"{np.max(nearest_distance):.6f}"
    )

    print()
    print("-" * 100)
    print("NEIGHBOUR STEERING")
    print("-" * 100)

    print(
        f"Mean neighbour steering:    "
        f"{np.mean(neighbor_mean):.6f}"
    )

    print(
        f"Mean neighbour std:         "
        f"{np.mean(neighbor_std):.6f}"
    )

    print(
        f"Mean neighbour min:         "
        f"{np.mean(neighbor_min):.6f}"
    )

    print(
        f"Mean neighbour max:         "
        f"{np.mean(neighbor_max):.6f}"
    )

    print()
    print("-" * 100)
    print("NEAR-ZERO NEIGHBOUR RATIO")
    print("-" * 100)

    print(
        f"Mean ratio:                 "
        f"{np.mean(near_zero_ratio):.4f}"
    )

    print(
        f"Median ratio:               "
        f"{np.median(near_zero_ratio):.4f}"
    )

    print(
        f"Samples with >= 50%:        "
        f"{np.mean(near_zero_ratio >= 0.50) * 100:.2f}%"
    )

    print(
        f"Samples with >= 75%:        "
        f"{np.mean(near_zero_ratio >= 0.75) * 100:.2f}%"
    )

    print(
        f"Samples with >= 90%:        "
        f"{np.mean(near_zero_ratio >= 0.90) * 100:.2f}%"
    )

    print()
    print("-" * 100)
    print("KNN PREDICTION")
    print("-" * 100)

    print(
        f"KNN MAE:                   "
        f"{np.mean(knn_error):.6f}"
    )

    print(
        f"Baseline MAE (predict 0):  "
        f"{np.mean(np.abs(y_query)):.6f}"
    )

    print(
        f"Improvement:                "
        f"{(1 - np.mean(knn_error) /
           np.mean(np.abs(y_query))) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # How many have dangerous neighbours?
    # ---------------------------------------------------------

    print()
    print("-" * 100)
    print("NON-ZERO / LARGE-STEERING NEIGHBOURS")
    print("-" * 100)

    for limit in [0.005, 0.01, 0.025, 0.05, 0.1]:

        ratio = np.mean(
            neighbor_abs_y >= limit,
            axis=1,
        )

        print(
            f"Mean neighbours with |steer| >= {limit:<5}: "
            f"{np.mean(ratio) * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Worst local ambiguity cases
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("MOST AMBIGUOUS NEAR-ZERO SAMPLES")
    print("=" * 100)

    # Sort by neighbour steering std
    worst_indices = np.argsort(
        neighbor_std
    )[::-1][:20]

    print(
        f"{'Test index':>12}"
        f"{'Actual':>12}"
        f"{'NN dist':>12}"
        f"{'NN mean':>12}"
        f"{'NN std':>12}"
        f"{'Near-zero %':>14}"
        f"{'NN min':>12}"
        f"{'NN max':>12}"
    )

    print("-" * 100)

    for i in worst_indices:

        print(
            f"{selected_indices[i]:>12}"
            f"{y_query[i]:>12.6f}"
            f"{nearest_distance[i]:>12.6f}"
            f"{neighbor_mean[i]:>12.6f}"
            f"{neighbor_std[i]:>12.6f}"
            f"{near_zero_ratio[i] * 100:>13.1f}%"
            f"{neighbor_min[i]:>12.6f}"
            f"{neighbor_max[i]:>12.6f}"
        )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        """
If near-zero test samples have mostly near-zero neighbours,
the feature space contains useful information for predicting
near-zero steering.

If near-zero samples frequently have neighbours with large
positive/negative steering, the mapping from X -> steering
is locally ambiguous.

In that case, changing the neural network architecture alone
is unlikely to completely solve the problem.
"""
    )

    return {
        "selected_indices": selected_indices,
        "distances": distances,
        "neighbor_indices": indices,
        "neighbor_y": neighbor_y,
        "neighbor_mean": neighbor_mean,
        "neighbor_std": neighbor_std,
        "near_zero_ratio": near_zero_ratio,
        "knn_prediction": knn_prediction,
        "knn_error": knn_error,
    }

def analyze_near_zero_model_failures(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    near_zero_threshold=0.001,
    error_threshold=0.01,
    k=20,
    top_n=30,
):
    """
    Finds near-zero test samples where the model makes a large error,
    despite having locally similar near-zero samples in the training set.

    This helps distinguish:
        1. missing/ambiguous training data
        2. model approximation/training problems
    """

    print("=" * 100)
    print("NEAR-ZERO MODEL FAILURES WITH CONSISTENT TRAINING NEIGHBOURS")
    print("=" * 100)

    # Flatten prediction
    y_pred = np.asarray(y_pred).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)

    # ------------------------------------------------------------------
    # 1. Select near-zero test samples
    # ------------------------------------------------------------------

    near_zero_mask = np.abs(y_test) < near_zero_threshold

    test_indices = np.where(near_zero_mask)[0]

    print(f"Near-zero test samples: {len(test_indices)}")

    # ------------------------------------------------------------------
    # 2. Calculate model errors
    # ------------------------------------------------------------------

    errors = np.abs(y_test - y_pred)

    # Only investigate samples where model error is significant
    candidate_indices = test_indices[
        errors[test_indices] >= error_threshold
    ]

    print(
        f"Near-zero samples with error >= {error_threshold}: "
        f"{len(candidate_indices)}"
    )

    if len(candidate_indices) == 0:
        print("\nNo candidate failures found.")
        return

    # ------------------------------------------------------------------
    # 3. Fit KNN on TRAINING DATA ONLY
    # ------------------------------------------------------------------

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    knn.fit(X_train)

    distances, neighbour_indices = knn.kneighbors(
        X_test[candidate_indices]
    )

    # ------------------------------------------------------------------
    # 4. Analyse each candidate
    # ------------------------------------------------------------------

    results = []

    for row, test_idx in enumerate(candidate_indices):

        neighbour_idx = neighbour_indices[row]
        neighbour_dist = distances[row]

        neighbour_steering = y_train[neighbour_idx]

        near_zero_ratio = np.mean(
            np.abs(neighbour_steering) < near_zero_threshold
        )

        neighbour_mean = np.mean(neighbour_steering)
        neighbour_std = np.std(neighbour_steering)

        results.append({
            "test_index": test_idx,
            "actual": y_test[test_idx],
            "prediction": y_pred[test_idx],
            "error": errors[test_idx],
            "nearest_distance": neighbour_dist[0],
            "neighbour_mean": neighbour_mean,
            "neighbour_std": neighbour_std,
            "near_zero_ratio": near_zero_ratio,
            "neighbour_min": np.min(neighbour_steering),
            "neighbour_max": np.max(neighbour_steering),
        })

    # ------------------------------------------------------------------
    # 5. Sort by strongest model failure
    # ------------------------------------------------------------------

    results.sort(
        key=lambda x: x["error"],
        reverse=True
    )

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------

    print("\n" + "-" * 100)
    print("TOP MODEL FAILURES")
    print("-" * 100)

    print(
        f"{'Index':>10} "
        f"{'Actual':>10} "
        f"{'Prediction':>12} "
        f"{'Error':>10} "
        f"{'NN dist':>10} "
        f"{'NN mean':>10} "
        f"{'NN std':>10} "
        f"{'NZ ratio':>10}"
    )

    print("-" * 100)

    for r in results[:top_n]:

        print(
            f"{r['test_index']:10d} "
            f"{r['actual']:10.6f} "
            f"{r['prediction']:12.6f} "
            f"{r['error']:10.6f} "
            f"{r['nearest_distance']:10.6f} "
            f"{r['neighbour_mean']:10.6f} "
            f"{r['neighbour_std']:10.6f} "
            f"{r['near_zero_ratio'] * 100:9.1f}%"
        )

    # ------------------------------------------------------------------
    # 7. Identify "strong evidence" cases
    # ------------------------------------------------------------------

    strong_cases = [
        r for r in results
        if (
            r["near_zero_ratio"] >= 0.90
            and r["neighbour_std"] < 0.01
        )
    ]

    print("\n" + "=" * 100)
    print("STRONG MODEL-FAILURE CASES")
    print("=" * 100)

    print(
        "Criteria:"
        "\n  - >= 90% of neighbours are near-zero"
        "\n  - neighbour steering std < 0.01"
        f"\n  - model error >= {error_threshold}"
    )

    print(f"\nCases found: {len(strong_cases)}")

    if strong_cases:

        print("\nThese are particularly interesting because the local")
        print("training data is highly consistent, but the model still")
        print("produces a large prediction error.")

        print("\n" + "-" * 100)

        for r in strong_cases[:top_n]:

            print(
                f"Index={r['test_index']:7d} | "
                f"actual={r['actual']: .6f} | "
                f"pred={r['prediction']: .6f} | "
                f"error={r['error']: .6f} | "
                f"NN dist={r['nearest_distance']: .6f} | "
                f"NN mean={r['neighbour_mean']: .6f} | "
                f"NN std={r['neighbour_std']: .6f} | "
                f"NZ={r['near_zero_ratio'] * 100:5.1f}%"
            )

    # ------------------------------------------------------------------
    # 8. Summary statistics
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"Candidate failures:       {len(results)}")
    print(f"Strong model failures:    {len(strong_cases)}")

    if results:

        print(
            f"Mean candidate error:     "
            f"{np.mean([r['error'] for r in results]):.6f}"
        )

        print(
            f"Mean NN distance:         "
            f"{np.mean([r['nearest_distance'] for r in results]):.6f}"
        )

        print(
            f"Mean NN near-zero ratio:  "
            f"{np.mean([r['near_zero_ratio'] for r in results]) * 100:.2f}%"
        )

    return results

def inspect_model_forward_pass(
    model,
    X_test,
    y_test,
    y_pred,
    sample_idx: int
) -> None:
    """
    Inspect the forward pass of a trained Keras model for one test sample.

    Prints:
      - actual target
      - final model prediction
      - absolute error
      - input features
      - output statistics of every model layer

    This is useful for identifying where an unexpectedly large
    prediction starts to appear inside the neural network.
    """

    # ------------------------------------------------------------------
    # Input sample
    # ------------------------------------------------------------------
    x = X_test[sample_idx:sample_idx + 1]

    # y_test has shape (N,)
    actual = float(y_test[sample_idx])

    # y_pred has shape (N, 1)
    prediction = float(y_pred[sample_idx, 0])

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 90)
    print(f"FORWARD PASS INSPECTION - SAMPLE {sample_idx}")
    print("=" * 90)

    print(f"Actual:       {actual: .6f}")
    print(f"Final pred:   {prediction: .6f}")
    print(f"Absolute err: {abs(actual - prediction): .6f}")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    print("\nINPUT")
    print("-" * 90)

    print("Shape:", x.shape)
    print("Values:")
    print(x)

    # ------------------------------------------------------------------
    # Build model that returns every layer output
    # ------------------------------------------------------------------
    layer_outputs = [layer.output for layer in model.layers]

    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=layer_outputs
    )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    outputs = activation_model.predict(
        x,
        verbose=0
    )

    # If the model has only one layer output, Keras may return an array
    # instead of a list. Normalize it to a list.
    if not isinstance(outputs, list):
        outputs = [outputs]

    # ------------------------------------------------------------------
    # Layer outputs
    # ------------------------------------------------------------------
    print("\nLAYER OUTPUTS")
    print("-" * 90)

    for layer, output in zip(model.layers, outputs):

        output_array = np.asarray(output)

        print(f"\nLayer: {layer.name}")
        print(f"Type:  {layer.__class__.__name__}")
        print(f"Shape: {output_array.shape}")

        print(
            f"Min:  {output_array.min(): .6f} | "
            f"Max:  {output_array.max(): .6f} | "
            f"Mean: {output_array.mean(): .6f} | "
            f"Std:  {output_array.std(): .6f}"
        )

        print("Values:")
        print(output_array.flatten())

    # ------------------------------------------------------------------
    # Final consistency check
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL OUTPUT CHECK")
    print("=" * 90)

    print(f"Expected prediction: {prediction: .6f}")

    final_output = np.asarray(outputs[-1]).flatten()[0]

    print(f"Forward-pass output: {final_output: .6f}")
    print(
        f"Difference:          "
        f"{abs(prediction - final_output): .12f}"
    )

    print("=" * 90)

def inspect_output_layer(model) -> None:
    """
    Inspect the weights and bias of the final output layer.
    """

    output_layer = model.layers[-1]

    weights, bias = output_layer.get_weights()

    print("=" * 90)
    print("OUTPUT LAYER WEIGHTS")
    print("=" * 90)

    print("Layer:", output_layer.name)
    print("Weights shape:", weights.shape)
    print("Bias shape:", bias.shape)

    print("\nWeights:")
    print(weights.flatten())

    print("\nBias:")
    print(bias)

    print("\nWeight statistics:")
    print(f"Min:  {weights.min(): .6f}")
    print(f"Max:  {weights.max(): .6f}")
    print(f"Mean: {weights.mean(): .6f}")
    print(f"Std:  {weights.std(): .6f}")

    print("\nBias:")
    print(bias)

def analyze_near_zero_model_failures(
    X_test,
    y_test,
    y_pred,
    X_train,
    y_train,
    threshold=0.001,
    error_threshold=0.01,
    k=20,
    max_samples=None,
):
    """
    Analyze near-zero steering samples and compare model predictions
    against the local training-data neighbourhood.

    The goal is to determine whether large model errors occur even when
    the corresponding region of the training dataset contains highly
    consistent near-zero steering targets.
    """

    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    print("=" * 100)
    print("NEAR-ZERO MODEL BEHAVIOUR ANALYSIS")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Normalize prediction shape
    # ------------------------------------------------------------------

    y_test_flat = np.asarray(y_test).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)
    y_train_flat = np.asarray(y_train).reshape(-1)

    # ------------------------------------------------------------------
    # Select near-zero test samples
    # ------------------------------------------------------------------

    near_zero_mask = np.abs(y_test_flat) < threshold
    near_zero_indices = np.where(near_zero_mask)[0]

    if max_samples is not None:
        near_zero_indices = near_zero_indices[:max_samples]

    print(f"\nThreshold:             |steering| < {threshold}")
    print(f"Near-zero test samples: {np.sum(near_zero_mask)}")
    print(f"Samples analyzed:       {len(near_zero_indices)}")
    print(f"K:                      {k}")

    # ------------------------------------------------------------------
    # Build KNN model on training data
    # ------------------------------------------------------------------

    knn = NearestNeighbors(n_neighbors=k)

    knn.fit(X_train)

    distances, neighbour_indices = knn.kneighbors(
        X_test[near_zero_indices]
    )

    # ------------------------------------------------------------------
    # Calculate local statistics
    # ------------------------------------------------------------------

    neighbour_targets = y_train_flat[neighbour_indices]

    neighbour_mean = np.mean(neighbour_targets, axis=1)
    neighbour_std = np.std(neighbour_targets, axis=1)

    neighbour_min = np.min(neighbour_targets, axis=1)
    neighbour_max = np.max(neighbour_targets, axis=1)

    nearest_distance = distances[:, 0]

    near_zero_ratio = np.mean(
        np.abs(neighbour_targets) < threshold,
        axis=1
    )

    # ------------------------------------------------------------------
    # Model errors
    # ------------------------------------------------------------------

    actual = y_test_flat[near_zero_indices]
    prediction = y_pred_flat[near_zero_indices]

    errors = np.abs(actual - prediction)

    # ------------------------------------------------------------------
    # Find failures
    # ------------------------------------------------------------------

    failure_mask = errors >= error_threshold

    failure_indices = np.where(failure_mask)[0]

    print("\n" + "-" * 100)
    print("FAILURE SUMMARY")
    print("-" * 100)

    print(f"Failures with error >= {error_threshold}: "
          f"{len(failure_indices)}")

    print(
        f"Failure percentage: "
        f"{100 * len(failure_indices) / len(near_zero_indices):.2f}%"
    )

    # ------------------------------------------------------------------
    # Strong failures
    #
    # Consistent local training neighbourhood:
    #   >= 90% near-zero
    #   neighbour std < 0.01
    #   model error >= threshold
    # ------------------------------------------------------------------

    strong_failure_mask = (
        (errors >= error_threshold)
        &
        (near_zero_ratio >= 0.90)
        &
        (neighbour_std < 0.01)
    )

    strong_failure_indices = np.where(
        strong_failure_mask
    )[0]

    print("\n" + "-" * 100)
    print("STRONG MODEL FAILURES")
    print("-" * 100)

    print("Criteria:")
    print("  error >= %.3f" % error_threshold)
    print("  >= 90%% near-zero neighbours")
    print("  neighbour steering std < 0.01")

    print(f"\nCases found: {len(strong_failure_indices)}")

    # ------------------------------------------------------------------
    # Sort failures by model error
    # ------------------------------------------------------------------

    sorted_failures = failure_indices[
        np.argsort(errors[failure_indices])[::-1]
    ]

    sorted_strong_failures = strong_failure_indices[
        np.argsort(errors[strong_failure_indices])[::-1]
    ]

    # ------------------------------------------------------------------
    # Print top failures
    # ------------------------------------------------------------------

    print("\n" + "-" * 100)
    print("TOP MODEL FAILURES")
    print("-" * 100)

    print(
        f"{'Index':>8} "
        f"{'Actual':>10} "
        f"{'Pred':>10} "
        f"{'Error':>10} "
        f"{'NN dist':>10} "
        f"{'NN mean':>10} "
        f"{'NN std':>10} "
        f"{'NZ ratio':>10}"
    )

    for local_idx in sorted_failures[:30]:

        original_idx = near_zero_indices[local_idx]

        print(
            f"{original_idx:8d} "
            f"{actual[local_idx]:10.6f} "
            f"{prediction[local_idx]:10.6f} "
            f"{errors[local_idx]:10.6f} "
            f"{nearest_distance[local_idx]:10.6f} "
            f"{neighbour_mean[local_idx]:10.6f} "
            f"{neighbour_std[local_idx]:10.6f} "
            f"{100 * near_zero_ratio[local_idx]:9.1f}%"
        )

    # ------------------------------------------------------------------
    # Print strong failures
    # ------------------------------------------------------------------

    print("\n" + "-" * 100)
    print("STRONG MODEL FAILURES - TOP 30")
    print("-" * 100)

    for local_idx in sorted_strong_failures[:30]:

        original_idx = near_zero_indices[local_idx]

        print(
            f"Index={original_idx:6d} | "
            f"actual={actual[local_idx]: .6f} | "
            f"pred={prediction[local_idx]: .6f} | "
            f"error={errors[local_idx]: .6f} | "
            f"NN dist={nearest_distance[local_idx]: .6f} | "
            f"NN mean={neighbour_mean[local_idx]: .6f} | "
            f"NN std={neighbour_std[local_idx]: .6f} | "
            f"NZ={100 * near_zero_ratio[local_idx]:.1f}%"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Mean near-zero MAE:       "
        f"{np.mean(errors):.6f}"
    )

    print(
        f"Median near-zero MAE:     "
        f"{np.median(errors):.6f}"
    )

    print(
        f"Mean prediction:          "
        f"{np.mean(prediction):.6f}"
    )

    print(
        f"Mean absolute prediction:"
        f" {np.mean(np.abs(prediction)):.6f}"
    )

    print(
        f"Mean NN distance:         "
        f"{np.mean(nearest_distance):.6f}"
    )

    print(
        f"Mean NN near-zero ratio:  "
        f"{np.mean(near_zero_ratio) * 100:.2f}%"
    )

    print(
        f"Strong model failures:    "
        f"{len(strong_failure_indices)}"
    )

    # ------------------------------------------------------------------
    # Return structured data
    # ------------------------------------------------------------------

    return {
        "indices": near_zero_indices,
        "actual": actual,
        "prediction": prediction,
        "error": errors,
        "nearest_distance": nearest_distance,
        "neighbour_mean": neighbour_mean,
        "neighbour_std": neighbour_std,
        "neighbour_min": neighbour_min,
        "neighbour_max": neighbour_max,
        "near_zero_ratio": near_zero_ratio,
        "failure_indices": near_zero_indices[failure_indices],
        "strong_failure_indices":
            near_zero_indices[strong_failure_indices],
    }
def inspect_output_layer_contributions(
    model,
    X_test,
    y_test,
    y_pred,
    sample_indices,
):
    """
    Inspect how the final Dense output layer produces the prediction.

    Assumes the model structure contains:
        Input -> ... -> Dense(16, activation=...) -> output Dense(1)

    For every requested sample, prints:
        - actual target
        - model prediction
        - output bias
        - activation of every neuron in the last hidden layer
        - output-layer weight
        - contribution of every neuron
        - sum of positive/negative contributions

    This is a diagnostic function only. It does not modify the model.
    """

    import numpy as np

    print("=" * 100)
    print("OUTPUT LAYER CONTRIBUTION ANALYSIS")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Normalize arrays
    # ------------------------------------------------------------------

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # ------------------------------------------------------------------
    # Find final Dense output layer
    # ------------------------------------------------------------------

    output_layer = model.layers[-1]

    weights = output_layer.get_weights()

    if len(weights) != 2:
        raise ValueError(
            "Expected final output layer to have kernel and bias."
        )

    output_weights = weights[0].reshape(-1)
    output_bias = float(np.asarray(weights[1]).reshape(-1)[0])

    # ------------------------------------------------------------------
    # Find last hidden layer
    # ------------------------------------------------------------------

    if len(model.layers) < 2:
        raise ValueError("Model does not contain a hidden layer.")

    hidden_layer = model.layers[-2]

    # Create model that returns hidden representation
    try:
        import tensorflow as tf

        hidden_model = tf.keras.Model(
            inputs=model.input,
            outputs=hidden_layer.output
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not create hidden-layer model."
        ) from exc

    # ------------------------------------------------------------------
    # Print output layer information
    # ------------------------------------------------------------------

    print("\nFINAL OUTPUT LAYER")
    print("-" * 100)

    print(f"Layer:          {output_layer.name}")
    print(f"Hidden layer:   {hidden_layer.name}")
    print(f"Weights shape:  {weights[0].shape}")
    print(f"Bias shape:     {weights[1].shape}")

    print("\nOutput weights:")
    print(output_weights)

    print(f"\nBias: {output_bias:.9f}")

    # ------------------------------------------------------------------
    # Inspect samples
    # ------------------------------------------------------------------

    for sample_idx in sample_indices:

        if sample_idx < 0 or sample_idx >= len(X_test):
            print(
                f"\nWARNING: sample index {sample_idx} "
                f"is outside X_test."
            )
            continue

        x = X_test[sample_idx:sample_idx + 1]

        actual = float(y_test[sample_idx])
        prediction = float(y_pred[sample_idx])

        # --------------------------------------------------------------
        # Forward pass through hidden layer
        # --------------------------------------------------------------

        hidden_output = np.asarray(
            hidden_model.predict(x, verbose=0)
        ).reshape(-1)

        # --------------------------------------------------------------
        # Make sure dimensions match
        # --------------------------------------------------------------

        if len(hidden_output) != len(output_weights):
            raise ValueError(
                f"Hidden output size ({len(hidden_output)}) does not "
                f"match output weight size ({len(output_weights)})."
            )

        # --------------------------------------------------------------
        # Calculate contributions
        # --------------------------------------------------------------

        contributions = hidden_output * output_weights

        positive_sum = np.sum(
            contributions[contributions > 0]
        )

        negative_sum = np.sum(
            contributions[contributions < 0]
        )

        reconstructed_prediction = (
            output_bias + np.sum(contributions)
        )

        # --------------------------------------------------------------
        # Header
        # --------------------------------------------------------------

        print("\n")
        print("=" * 100)
        print(f"SAMPLE {sample_idx}")
        print("=" * 100)

        print(f"Actual:                 {actual: .9f}")
        print(f"Model prediction:       {prediction: .9f}")
        print(f"Absolute error:         {abs(actual - prediction): .9f}")

        print("\nINPUT")
        print("-" * 100)

        print(x)

        # --------------------------------------------------------------
        # Contribution table
        # --------------------------------------------------------------

        print("\nHIDDEN → OUTPUT CONTRIBUTIONS")
        print("-" * 100)

        print(
            f"{'Neuron':>8} "
            f"{'Activation':>15} "
            f"{'Weight':>15} "
            f"{'Contribution':>18}"
        )

        print("-" * 100)

        # Sort by absolute contribution
        sorted_indices = np.argsort(
            np.abs(contributions)
        )[::-1]

        for neuron_idx in sorted_indices:

            print(
                f"{neuron_idx:8d} "
                f"{hidden_output[neuron_idx]:15.8f} "
                f"{output_weights[neuron_idx]:15.8f} "
                f"{contributions[neuron_idx]:18.8f}"
            )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        print("\nCONTRIBUTION SUMMARY")
        print("-" * 100)

        print(
            f"Bias contribution:             "
            f"{output_bias: .9f}"
        )

        print(
            f"Positive contributions:         "
            f"{positive_sum: .9f}"
        )

        print(
            f"Negative contributions:         "
            f"{negative_sum: .9f}"
        )

        print(
            f"Sum of neuron contributions:    "
            f"{np.sum(contributions): .9f}"
        )

        print(
            f"Reconstructed prediction:       "
            f"{reconstructed_prediction: .9f}"
        )

        print(
            f"Actual model prediction:        "
            f"{prediction: .9f}"
        )

        print(
            f"Reconstruction difference:      "
            f"{abs(reconstructed_prediction - prediction): .12f}"
        )

        # --------------------------------------------------------------
        # Dominant neurons
        # --------------------------------------------------------------

        top_n = min(5, len(contributions))

        print("\nTOP CONTRIBUTING NEURONS")
        print("-" * 100)

        for rank, neuron_idx in enumerate(
            sorted_indices[:top_n],
            start=1
        ):
            print(
                f"{rank}. neuron {neuron_idx:2d} | "
                f"activation={hidden_output[neuron_idx]: .8f} | "
                f"weight={output_weights[neuron_idx]: .8f} | "
                f"contribution={contributions[neuron_idx]: .8f}"
            )

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

def analyze_near_zero_hidden_neurons(
    model,
    x_data,
    y_data,
    near_zero_threshold: float = 0.001,
    failure_threshold: float = 0.01,
    strong_nz_ratio: float = 0.90,
    strong_nn_std: float = 0.01,
    max_failures: int = 1042,
    batch_size: int = 4096,
) -> None:
    """
    Analyze hidden-layer neuron activations for near-zero steering samples.

    The goal is to determine which hidden neurons are responsible for
    large prediction errors when the true steering value is approximately zero.

    The analysis compares:
        1. Normal near-zero samples
        2. Strong near-zero failures

    For each hidden neuron we report:
        - mean activation
        - std activation
        - percentage of active samples
        - maximum activation
        - mean absolute contribution to output

    Parameters
    ----------
    model:
        Trained Keras model.

    x_data:
        Input samples, shape (N, num_features).

    y_data:
        Ground-truth steering values, shape (N,) or (N, 1).

    near_zero_threshold:
        Samples with |steering| < this value are considered near-zero.

    failure_threshold:
        Prediction error threshold for a failure.

    strong_nz_ratio:
        Minimum near-zero neighbour ratio used to identify strong failures.

    strong_nn_std:
        Maximum neighbour steering std used to identify strong failures.

    max_failures:
        Maximum number of strong failure samples to analyze.

    batch_size:
        Batch size used for prediction.
    """

    import numpy as np
    import tensorflow as tf

    print("\n" + "=" * 100)
    print("NEAR-ZERO HIDDEN NEURON ACTIVATION ANALYSIS")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Prepare arrays
    # ------------------------------------------------------------------

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data).reshape(-1)

    if len(x_data) != len(y_data):
        raise ValueError(
            f"x_data and y_data have different lengths: "
            f"{len(x_data)} vs {len(y_data)}"
        )

    print(f"Samples:                {len(x_data)}")
    print(f"Input shape:            {x_data.shape}")
    print(f"Near-zero threshold:    {near_zero_threshold}")
    print(f"Failure threshold:      {failure_threshold}")

    # ------------------------------------------------------------------
    # Find hidden layers
    # ------------------------------------------------------------------

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    if len(dense_layers) < 2:
        raise ValueError(
            "Expected at least two Dense layers: "
            "hidden layer(s) + output layer."
        )

    output_layer = dense_layers[-1]
    hidden_layer = dense_layers[-2]

    print("\n" + "-" * 100)
    print("MODEL STRUCTURE")
    print("-" * 100)

    print(f"Hidden layer:           {hidden_layer.name}")
    print(f"Hidden units:           {hidden_layer.units}")

    print(f"Output layer:           {output_layer.name}")
    print(f"Output units:           {output_layer.units}")

    # ------------------------------------------------------------------
    # Build activation model
    # ------------------------------------------------------------------

    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=hidden_layer.output,
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    print("\nCalculating predictions...")

    predictions = model.predict(
        x_data,
        batch_size=batch_size,
        verbose=0,
    )

    predictions = np.asarray(predictions).reshape(-1)

    errors = np.abs(predictions - y_data)

    # ------------------------------------------------------------------
    # Near-zero samples
    # ------------------------------------------------------------------

    near_zero_mask = np.abs(y_data) < near_zero_threshold

    near_zero_indices = np.where(near_zero_mask)[0]

    print(f"Near-zero samples:      {len(near_zero_indices)}")

    if len(near_zero_indices) == 0:
        print("No near-zero samples found.")
        return

    # ------------------------------------------------------------------
    # Strong failures
    #
    # We reproduce the important part of the previous analysis:
    #
    #   actual approximately zero
    #   error >= threshold
    #
    # The neighbour information is optional here because the purpose of
    # this function is specifically hidden-neuron analysis.
    #
    # We still print the strong-failure population separately.
    # ------------------------------------------------------------------

    failure_mask = (
        near_zero_mask
        & (errors >= failure_threshold)
    )

    failure_indices = np.where(failure_mask)[0]

    print(f"Near-zero failures:     {len(failure_indices)}")

    if len(failure_indices) == 0:
        print("No near-zero failures found.")
        return

    # Sort by largest error
    failure_indices = failure_indices[
        np.argsort(errors[failure_indices])[::-1]
    ]

    if max_failures is not None:
        failure_indices = failure_indices[:max_failures]

    # ------------------------------------------------------------------
    # Define normal near-zero samples
    #
    # We deliberately exclude failures so that we compare:
    #
    #       GOOD near-zero
    #       vs
    #       BAD near-zero
    #
    # This makes the neuron differences much easier to interpret.
    # ------------------------------------------------------------------

    good_indices = near_zero_indices[
        errors[near_zero_indices] < failure_threshold
    ]

    print(f"Good near-zero samples: {len(good_indices)}")
    print(f"Failures analyzed:      {len(failure_indices)}")

    if len(good_indices) == 0:
        print("No good near-zero samples available.")
        return

    # ------------------------------------------------------------------
    # Calculate hidden activations
    # ------------------------------------------------------------------

    print("\nCalculating hidden-layer activations...")

    hidden_activations = activation_model.predict(
        x_data,
        batch_size=batch_size,
        verbose=0,
    )

    hidden_activations = np.asarray(hidden_activations)

    # Flatten any dimensions after sample dimension
    hidden_activations = hidden_activations.reshape(
        hidden_activations.shape[0],
        -1,
    )

    num_neurons = hidden_activations.shape[1]

    print(f"Activation matrix:     {hidden_activations.shape}")

    # ------------------------------------------------------------------
    # Extract GOOD / BAD activations
    # ------------------------------------------------------------------

    good_act = hidden_activations[good_indices]
    bad_act = hidden_activations[failure_indices]

    # ------------------------------------------------------------------
    # Output layer weights
    # ------------------------------------------------------------------

    output_weights, output_bias = output_layer.get_weights()

    output_weights = np.asarray(output_weights).reshape(-1)

    output_bias_value = float(np.asarray(output_bias).reshape(-1)[0])

    if len(output_weights) != num_neurons:
        raise ValueError(
            f"Hidden/output mismatch: "
            f"{num_neurons} hidden activations but "
            f"{len(output_weights)} output weights."
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    good_mean = np.mean(good_act, axis=0)
    good_std = np.std(good_act, axis=0)

    bad_mean = np.mean(bad_act, axis=0)
    bad_std = np.std(bad_act, axis=0)

    good_active = np.mean(good_act > 1e-8, axis=0) * 100.0
    bad_active = np.mean(bad_act > 1e-8, axis=0) * 100.0

    good_max = np.max(good_act, axis=0)
    bad_max = np.max(bad_act, axis=0)

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------

    good_contribution = good_act * output_weights
    bad_contribution = bad_act * output_weights

    good_mean_abs_contribution = np.mean(
        np.abs(good_contribution),
        axis=0,
    )

    bad_mean_abs_contribution = np.mean(
        np.abs(bad_contribution),
        axis=0,
    )

    bad_max_abs_contribution = np.max(
        np.abs(bad_contribution),
        axis=0,
    )

    # Difference in activation
    activation_difference = bad_mean - good_mean

    # ------------------------------------------------------------------
    # Neuron importance score
    #
    # We want neurons that:
    #
    #   1. activate more in BAD samples
    #   2. have a significant output weight
    #
    # This is not a mathematical causal score; it is a diagnostic ranking.
    # ------------------------------------------------------------------

    diagnostic_score = (
        np.abs(activation_difference)
        * np.abs(output_weights)
    )

    ranking = np.argsort(diagnostic_score)[::-1]

    # ------------------------------------------------------------------
    # Print main table
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("NEURON COMPARISON: GOOD vs BAD NEAR-ZERO SAMPLES")
    print("=" * 100)

    print(
        f"{'Neuron':>8} "
        f"{'Weight':>11} "
        f"{'GOOD mean':>12} "
        f"{'BAD mean':>12} "
        f"{'Delta':>12} "
        f"{'GOOD std':>11} "
        f"{'BAD std':>11} "
        f"{'GOOD act%':>11} "
        f"{'BAD act%':>10} "
        f"{'BAD |C|':>12}"
    )

    print("-" * 100)

    for i in range(num_neurons):
        print(
            f"{i:8d} "
            f"{output_weights[i]:11.6f} "
            f"{good_mean[i]:12.6f} "
            f"{bad_mean[i]:12.6f} "
            f"{activation_difference[i]:12.6f} "
            f"{good_std[i]:11.6f} "
            f"{bad_std[i]:11.6f} "
            f"{good_active[i]:10.2f}% "
            f"{bad_active[i]:9.2f}% "
            f"{bad_mean_abs_contribution[i]:12.6f}"
        )

    # ------------------------------------------------------------------
    # Top diagnostic neurons
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("TOP DIAGNOSTIC NEURONS")
    print("=" * 100)

    print(
        "Ranking is based on:\n"
        "    |BAD mean activation - GOOD mean activation| "
        "* |output weight|\n"
    )

    print(
        f"{'Rank':>6} "
        f"{'Neuron':>8} "
        f"{'Weight':>11} "
        f"{'GOOD':>12} "
        f"{'BAD':>12} "
        f"{'Delta':>12} "
        f"{'Score':>12}"
    )

    print("-" * 80)

    for rank, neuron in enumerate(ranking, start=1):

        print(
            f"{rank:6d} "
            f"{neuron:8d} "
            f"{output_weights[neuron]:11.6f} "
            f"{good_mean[neuron]:12.6f} "
            f"{bad_mean[neuron]:12.6f} "
            f"{activation_difference[neuron]:12.6f} "
            f"{diagnostic_score[neuron]:12.6f}"
        )

    # ------------------------------------------------------------------
    # Top contributors specifically
    # ------------------------------------------------------------------

    contribution_ranking = np.argsort(
        bad_mean_abs_contribution
    )[::-1]

    print("\n" + "=" * 100)
    print("TOP NEURONS BY ABSOLUTE OUTPUT CONTRIBUTION")
    print("=" * 100)

    print(
        f"{'Rank':>6} "
        f"{'Neuron':>8} "
        f"{'Weight':>11} "
        f"{'BAD mean act':>14} "
        f"{'BAD mean |C|':>14} "
        f"{'BAD max |C|':>14}"
    )

    print("-" * 85)

    for rank, neuron in enumerate(contribution_ranking, start=1):

        print(
            f"{rank:6d} "
            f"{neuron:8d} "
            f"{output_weights[neuron]:11.6f} "
            f"{bad_mean[neuron]:14.6f} "
            f"{bad_mean_abs_contribution[neuron]:14.6f} "
            f"{bad_max_abs_contribution[neuron]:14.6f}"
        )

    # ------------------------------------------------------------------
    # Show individual worst failures
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("WORST FAILURES AND THEIR DOMINANT NEURONS")
    print("=" * 100)

    num_examples = min(20, len(failure_indices))

    for position in range(num_examples):

        idx = failure_indices[position]

        activation = hidden_activations[idx]

        contributions = activation * output_weights

        contribution_order = np.argsort(
            np.abs(contributions)
        )[::-1]

        top_neuron = contribution_order[0]

        print(
            f"\nIndex={idx}"
            f" | actual={y_data[idx]: .6f}"
            f" | pred={predictions[idx]: .6f}"
            f" | error={errors[idx]: .6f}"
        )

        print(
            f"  Dominant neuron: {top_neuron}"
            f" | activation={activation[top_neuron]: .6f}"
            f" | weight={output_weights[top_neuron]: .6f}"
            f" | contribution={contributions[top_neuron]: .6f}"
        )

        print("  Top 5 contributions:")

        for neuron in contribution_order[:5]:

            print(
                f"      neuron {neuron:2d}"
                f" | activation={activation[neuron]: .6f}"
                f" | weight={output_weights[neuron]: .6f}"
                f" | contribution={contributions[neuron]: .6f}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"Good near-zero samples:       {len(good_indices)}")
    print(f"Strong/worst samples:         {len(failure_indices)}")

    print("\nMost suspicious neurons:")

    for rank, neuron in enumerate(ranking[:5], start=1):

        print(
            f"  {rank}. neuron {neuron}"
            f" | weight={output_weights[neuron]: .6f}"
            f" | GOOD={good_mean[neuron]: .6f}"
            f" | BAD={bad_mean[neuron]: .6f}"
            f" | delta={activation_difference[neuron]: .6f}"
            f" | diagnostic score={diagnostic_score[neuron]: .6f}"
        )

    print("\nOutput bias:")
    print(f"  {output_bias_value: .9f}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    (
        train,
        validation,
        test,
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        scaler,
    ) = main()

    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    evaluate_model(
        model,
        X_test,
        y_test,
    )

    analyze_model_predictions(
        model,
        X_test,
        y_test,
    )
    y_pred = model.predict(X_test, verbose=1)

    analyze_steering_predictions(
        y_test,
        y_pred
    )
    plot_steering_timeseries(y_test, y_pred, num_samples=5000)
    plot_steering_timeseries(y_test, y_pred)

    analyze_steering_lag(
        y_test,
        y_pred,
        max_lag=50,
    )

    show_worst_steering_predictions(
        y_test,
        y_pred,
        n=100,
    )

    errors = np.abs(y_test.flatten() - y_pred.flatten())

    worst_indices = np.argsort(errors)[-5:][::-1]

    plot_prediction_context(
        y_test.flatten(),
        y_pred.flatten(),
        worst_indices,
        context=50
    )
    print("======================= ANDRAZ ===========================")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)
    print(type(X_test))
    print(type(y_test))
    print(type(y_pred))
    inspect_prediction_context(
        X_test,
        y_test,
        y_pred,
        sample_idx=387842,
        context=10
    )

    inspect_input_context(
        X_test,
        y_test,
        y_pred,
        sample_idx=387842,
        context=3
    )

    X_test_original = scaler.inverse_transform(X_test)

    inspect_original_input_context(
        X_test,
        y_test,
        y_pred,
        scaler,
        sample_idx=387842,
        context=5
    )

    inspect_nearest_training_samples(
        X_train,
        y_train,
        X_test,
        y_test,
        sample_idx=387842,
        n_neighbors=20,
    )
    analyze_knn_context(
        X_train,
        y_train,
        X_test,
        y_test,
        y_pred,
        sample_idx=387842,
    )

    analyze_near_zero_predictions(
        y_test,
        y_pred,
    )

    feature_names = [
        "speed",
        "acceleration",
        "distance",
        "lane_offset",
        "heading_error",
    ]

    analyze_near_zero_features(
        X_test,
        y_test,
        feature_names,
        zero_threshold=0.001,
    )

    correlations = analyze_feature_steering_relationship(
        X_test,
        y_test,
        feature_names,
    )

    effect_sizes = analyze_near_zero_separability(
        X_test,
        y_test,
        feature_names,
        threshold=0.001,
    )

    knn_analysis = analyze_near_zero_knn(
        X_train,
        y_train,
        X_test,
       y_test,
        threshold=0.001,
        k=20,
        n_samples=1000,
    )



    inspect_model_forward_pass(
        model,
        X_test,
        y_test,
        y_pred,
        sample_idx=434559
    )
    inspect_output_layer(model)
    analysis = analyze_near_zero_model_failures(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        X_train=X_train,
        y_train=y_train,
        threshold=0.001,
        error_threshold=0.01,
        k=20,
        max_samples=None,
    )

    inspect_output_layer_contributions(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        sample_indices=[
            379722,
            350718,
            437978,
            394358,
            311527,
        ],
    )
    analyze_near_zero_hidden_neurons(
        model=model,
        x_data=X_test,
        y_data=y_test,
    )