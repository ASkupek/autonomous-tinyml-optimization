import pandas as pd
from config import GLOBAL_CONFIG
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

@tf.keras.utils.register_keras_serializable()
class SmoothHuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta=0.01, smoothness_weight=0.002, sign_penalty_weight=0.03, name="smooth_huber_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.smoothness_weight = smoothness_weight

        self.huber = tf.keras.losses.Huber(delta=self.delta)

    def call(self, y_true, y_pred):
        base_loss = self.huber(y_true, y_pred)
        
        # 1. Kazen za napačen predznak (če je dejanski zasuk večji od 0.02 in model obrne volan v napačno smer)
        significant_mask = tf.cast(tf.abs(y_true) > 0.02, tf.float32)
        sign_mismatch = tf.cast(tf.sign(y_pred) != tf.sign(y_true), tf.float32)
        
        # 2. Časovna zveznost (jerk reduction)
        diffs = y_pred[1:] - y_pred[:-1]
        smoothness_loss = tf.reduce_mean(tf.square(diffs))
        
        return base_loss + (self.smoothness_weight * smoothness_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            "smoothness_weight": self.smoothness_weight
        })
        return config
def sort_dataset():
    print("Loading dataset")
    dataset = pd.read_csv(GLOBAL_CONFIG.pipeline.csv_path)

    dataset_sorted = (
        dataset.sort_values(by=["vehicle_id", "timestamp"])
          .reset_index(drop=True)
    )

    dataset_sorted.to_csv(GLOBAL_CONFIG.pipeline.path_to_downsampled_dataset, index=False)


def separate_dataset(dataset):
    unique_vehicles = sorted(dataset['vehicle_id'].unique())
    groups = np.array_split(unique_vehicles, 4)

    train_dataset = dataset[dataset['vehicle_id'].isin(np.concatenate([groups[0], groups[1]]))]
    validation_dataset = dataset[dataset['vehicle_id'].isin(groups[2])]
    test_dataset = dataset[dataset['vehicle_id'].isin(groups[3])]

    return (train_dataset, validation_dataset, test_dataset)

def scale_features(train_dataset, validation_dataset, test_dataset, input_features, output_feature):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_dataset[input_features])
    x_validation = scaler.transform(validation_dataset[input_features])
    X_test = scaler.transform(test_dataset[input_features])

    y_train = train_dataset[output_feature]
    y_validation = validation_dataset[output_feature]
    y_test = test_dataset[output_feature]

    return (scaler, X_train, y_train, x_validation, y_validation, X_test, y_test)

def scale_and_create_windows(train_dataset, validation_dataset, test_dataset, input_features, output_feature, window_size=10):
    scaler = StandardScaler()

    train_df = train_dataset.copy()
    val_df = validation_dataset.copy()
    test_df = test_dataset.copy()

    train_df[input_features] = scaler.fit_transform(train_dataset[input_features])
    val_df[input_features] = scaler.transform(validation_dataset[input_features])
    test_df[input_features] = scaler.transform(test_dataset[input_features])

    def build_windows(df):
        X_list, y_list = [], []
        for vehicle_id, group in df.groupby('vehicle_id'):
            group_sorted = group.sort_values(by="timestamp")
            features = group_sorted[input_features].to_numpy()
            targets = group_sorted[output_feature].to_numpy()
            
            for i in range(window_size - 1, len(group_sorted)):
                window = features[i - window_size + 1: i + 1]
                X_list.append(window.flatten()) # Oblika: (window_size * len(input_features),)
                y_list.append(targets[i])
                
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    X_train, y_train = build_windows(train_df)
    X_val, y_val = build_windows(val_df)
    X_test, y_test = build_windows(test_df)

    return scaler, X_train, y_train, X_val, y_val, X_test, y_test

def output_dataset(dataset, output_features):
    print(
            dataset[output_features]
            .describe()
        )
    #Detailed mount of data for example:
    # 1% only reporeent values lower thant -0.2xxx
    # 3% of data is lower than -0.096 etc.
    print(
        dataset[output_features]
        .quantile(
            [
                0.01,
                0.03,
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
    
    #we take all values and make them positive, then we eparate by bins so:
    # all classes between some steer
    # we put each element into dedicated class and we output for each group
    abs_steer = dataset[output_features].abs()
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

def output_specifc_set(dataset, output_features, input_features):
    # we calculate the dmean std min and max for each dataset train, test, validation
    features = input_features + [output_features]
    print(
        dataset[features]
        .agg(["count", "mean", "std", "min", "max"])
        .to_string()
    )

    # we calculate baseline so basically if we would predict 0 for everything and calculate mae
    baseline_mae = (
        dataset[output_features]
        .abs()
        .mean()
    )

    print(
        "Steer MAE baseline "
        "(predict 0): "
        f"{baseline_mae:.6f}"
    )

def quantiles_output(dataset, quantiles, input_features):
    print("\n===== FEATURE QUANTILES =====")
    print(input_features)
    for feature in input_features:
    
        print(f"\n--- {feature} ---")
    
        print(
            dataset[feature]
            .quantile(quantiles)
            .to_string()
        )

def corelation_representation(dataset, input_features, output_features):
    columns = input_features + [output_features]
    print(
        "\n===== CORRELATION MATRIX ====="
    )

    print(
        dataset[columns]
        .corr()
        .round(4)
        .to_string()
    )

def create_windows_for_scaled_data(train_dataset, validation_dataset, test_dataset, scaler, input_features, output_feature, window_size=10):
    train_df = train_dataset.copy()
    val_df = validation_dataset.copy()
    test_df = test_dataset.copy()

    train_df[input_features] = scaler.transform(train_dataset[input_features])
    val_df[input_features] = scaler.transform(validation_dataset[input_features])
    test_df[input_features] = scaler.transform(test_dataset[input_features])

    def build_windows(df):
        X_list, y_list = [], []
        for vehicle_id, group in df.groupby('vehicle_id'):
            group_sorted = group.sort_values(by="timestamp")
            features = group_sorted[input_features].to_numpy()
            targets = group_sorted[output_feature].to_numpy()
            
            for i in range(window_size - 1, len(group_sorted)):
                window = features[i - window_size + 1: i + 1]
                X_list.append(window.flatten())
                y_list.append(targets[i])
                
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    X_train_w, y_train_w = build_windows(train_df)
    X_val_w, y_val_w = build_windows(val_df)
    X_test_w, y_test_w = build_windows(test_df)

    return X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w
@tf.keras.utils.register_keras_serializable()
def steering_weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)

    abs_y = tf.abs(y_true)

    weights = 1.0 + 2.0 * tf.clip_by_value(abs_y / 0.1, 0.0, 1.0)

    return tf.reduce_mean(weights * error)

def train_and_save_model(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nBuild basic model")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        tf.keras.layers.Dense(97, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(101, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(58, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="tanh", name="steer")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=steering_weighted_mse,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    #sample_weights = 1.0 + (np.abs(y_train) ** 1.5) * 10.0
    model.fit(
        X_train, y_train,
        #sample_weight=sample_weights,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        verbose=1
    )
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    test_mae = np.mean(np.abs(y_test - y_pred))
    baseline_mae = np.mean(np.abs(y_test))
    
    print("\n" + "=" * 50)
    print("Final results on test set")
    print("=" * 50)
    print(f"Test MAE model:  {test_mae:.6f}")
    print(f"Baseline MAE (0): {baseline_mae:.6f}")
    print("=" * 50)
    
    model_path = "carla_mlp_controller.keras"
    model.save(model_path)
    print(f"Moded saved {model_path}")
    
    return model

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

def plot_actual_vs_predicted(model, X_test, y_test, max_samples=5000):
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test[:max_samples], label="Dejanski steer (Actual)", color="blue", alpha=0.7, linewidth=1.5)
    plt.plot(y_pred[:max_samples], label="Napovedani steer (Predicted)", color="red", linestyle="--", alpha=0.8, linewidth=1.5)
    
    plt.title("Primerjava dejanskih in napovedanih vrednosti krmila", fontsize=14)
    plt.xlabel("Časovni koraki (Sample Index)", fontsize=12)
    plt.ylabel("Krmilni kot (Steer)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

def check_steering_consistency(X_test, y_test, model, window_size=10, scaler=None):
    """
    Preveri skladnost med dejanskim krmilom, napovedjo in vhodnimi 
    značilnostmi (lane_offset, heading_error), da odkrije napačne predznake.
    """
    print("\n===== DIAGNOSTIKA PREDZNAKOV IN NESKLADIJ =====")
    
    # Izračun napovedi
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Ker je X_test sploščen (shape: N x 40), 
    # vzamemo zadnji časovni korak okna za vpogled v trenutno stanje (lane_offset in heading_error).
    # Glede na tvojo obliko 40-dimenzionalnega vhoda (10 korakov * 4 značilnosti: speed, acc, lane_offset, heading_error)
    # je zadnji korak na indeksih od -4 do -1.
    
    reshaped_X = X_test.reshape(-1, window_size, 4)
    last_step_features = reshaped_X[:, -1, :] # Zadnji korak okna
    last_step_features = scaler.inverse_transform(last_step_features)
    lane_offsets = last_step_features[:, 2]  # Indeks 2 je lane_offset
    heading_errors = last_step_features[:, 3] # Indeks 3 je heading_error
    
    mismatch_count = 0
    sign_flips = 0
    
    for i in range(len(y_test)):
        actual = y_test[i]
        pred = y_pred[i]
        l_off = lane_offsets[i]
        h_err = heading_errors[i]
        
        # 1. Preverjanje zamenjave predznaka med napovedjo in dejanskim stanjem
        if abs(actual) > 0.02 and abs(pred) > 0.02:
            if np.sign(actual) != np.sign(pred):
                sign_flips += 1
                
        # 2. Preverjanje fizikalne logike (npr. lane_offset in steer morata biti usklajena)
        # Če je lane_offset močno pozitiven, bi moral biti steer v določeni smeri
        if abs(l_off) > 0.5 and abs(actual) > 0.02:
            if np.sign(l_off) == np.sign(actual): # Odvisno od vaše konvencije v CARLA-i
                pass

    print(f"Skupaj testnih vzorcev: {len(y_test)}")
    print(f"Število napačnih predznakov (kjer se napoved in realnost na večjem ovinku razlikujeta): {sign_flips}")
    
    # Prikaz vzorčnih neskladij za lažjo debugiranje
    df_check = pd.DataFrame({
        'lane_offset': lane_offsets,
        'heading_error': heading_errors,
        'actual_steer': y_test,
        'pred_steer': y_pred
    })
    
    # Poiščimo primere z največjo napako
    df_check['error'] = np.abs(df_check['actual_steer'] - df_check['pred_steer'])
    worst_cases = df_check.sort_values(by='error', ascending=False).head(10)
    
    print("\nTop 10 najslabših napovedi (za analizo okoliščin):")
    print(worst_cases[['lane_offset', 'heading_error', 'actual_steer', 'pred_steer', 'error']].to_string())

def evaluate_steering_classes(y_true, y_pred):
    abs_true = np.abs(y_true)
    
    classes = [
        ("Ravnina (<= 0.02)", abs_true <= 0.02),
        ("Blagi zavoji (0.02 - 0.1)", (abs_true > 0.02) & (abs_true <= 0.1)),
        ("Ostri zavoji (0.1 - 0.3)", (abs_true > 0.1) & (abs_true <= 0.3)),
        ("Ekstremni zavoji (> 0.3)", abs_true > 0.3)
    ]
    
    metrics = {}
    print("--- Analiza napak krmila po razredih ---")
    for name, mask in classes:
        count = np.sum(mask)
        if count > 0:
            class_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            metrics[name] = {"MAE": class_mae, "Count": count}
            print(f"{name}: MAE = {class_mae:.5f} (vzorcev: {count})")
        else:
            metrics[name] = {"MAE": 0.0, "Count": 0}
            print(f"{name}: Brez vzorcev")
            
    return metrics

def load_processed_datasets(path="processed_dataset"):
    print("\n===== LOADING PROCESSED DATASET =====")

    train = np.load(os.path.join(path, "train.npz"))
    validation = np.load(os.path.join(path, "validation.npz"))
    test = np.load(os.path.join(path, "test.npz"))

    X_train = train["X"]
    y_train = train["y"]

    X_val = validation["X"]
    y_val = validation["y"]

    X_test = test["X"]
    y_test = test["y"]

    print(f"Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Test:       X={X_test.shape}, y={y_test.shape}")

    print("\n===== DATASET STATISTICS =====")

    print(
        f"Train y:      min={y_train.min():.6f}, "
        f"max={y_train.max():.6f}, "
        f"mean={y_train.mean():.6f}, "
        f"std={y_train.std():.6f}"
    )

    print(
        f"Validation y: min={y_val.min():.6f}, "
        f"max={y_val.max():.6f}, "
        f"mean={y_val.mean():.6f}, "
        f"std={y_val.std():.6f}"
    )

    print(
        f"Test y:       min={y_test.min():.6f}, "
        f"max={y_test.max():.6f}, "
        f"mean={y_test.mean():.6f}, "
        f"std={y_test.std():.6f}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    #if not os.path.exists(GLOBAL_CONFIG.pipeline.path_to_downsampled_dataset):
    #    cleaned_df = sort_dataset()

    #dataset = pd.read_csv(GLOBAL_CONFIG.pipeline.path_to_downsampled_dataset)
    # Since we saw in the analyis that we have some outliers

    #dataset = dataset[
    #    (dataset['lane_offset'].abs() <= 1.5) & 
    #    (dataset['heading_error'].abs() <= 1.0) &
    #    (dataset['speed'] >= 0.0)
    #]
    #(train_dataset, validation_dataset, test_dataset) = separate_dataset(dataset=dataset)

    #input_features = ['speed', 'acceleration', 'lane_offset', 'heading_error']
    #output_features = 'steer'

    #(scaler, X_train, y_train, x_validation, y_validation, X_test, y_test) = scale_features(train_dataset=train_dataset,
    #                                                                                       validation_dataset=validation_dataset, 
    #                                                                                        test_dataset=test_dataset, 
    #                                                                                        input_features=input_features, 
    #                                                                                        output_feature=output_features)
    #print("Dataset data representation")
    #output_dataset(dataset=dataset, output_features=output_features)
    #print("Train dataset data representation")
    #output_specifc_set(dataset=train_dataset, output_features=output_features, input_features=input_features)
    #print("Validation dataset data representation")
    #output_specifc_set(dataset=validation_dataset, output_features=output_features, input_features=input_features)
    #print("Test dataset data representation")
   # output_specifc_set(dataset=test_dataset, output_features=output_features, input_features=input_features)
   # print("Quantile for train dataset")
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
    #quantiles_output(dataset=train_dataset, quantiles=quantiles, input_features=input_features)
    #corelation_representation(dataset=train_dataset, input_features=input_features, output_features=output_features)

    #print("Now we move to scaled dataset")
    #print(X_train)
    #train_scaled = pd.DataFrame(
    #    X_train,
    #    columns=input_features
    #)
    #print(
    #    train_scaled
    #    .describe()
    #    .loc[
    #        ["mean", "std", "min", "max"]
    #    ]
    #)

    #print("Scaler statistic:")
    #for feature, mean, scale in zip(
    #    input_features,
    #    scaler.mean_,
    #    scaler.scale_,
    #):
    
    #    print(
    #        f"{feature:20s} "
    #        f"mean={mean:.8f} "
    #        f"scale={scale:.8f}"
    #    )
    #print("From here we try to build dataset with windows, so separate everything to windows")
    #X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w = create_windows_for_scaled_data(train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset, scaler=scaler, input_features=input_features, output_feature=output_features, window_size=10)
    #print("X_train window")
    #print(X_train_w)
    X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w = load_processed_datasets("processed_dataset")

    print("\n===== INPUT CHECK =====")
    print("X_train shape:", X_train_w.shape)
    print("y_train shape:", y_train_w.shape)
    print("X_val shape:", X_val_w.shape)
    print("y_val shape:", y_val_w.shape)
    print("X_test shape:", X_test_w.shape)
    print("y_test shape:", y_test_w.shape)

    print("\nFirst training sample:")
    print(X_train_w[0])

    print("\nFirst training target:")
    print(y_train_w[0])

    print("\nFirst test sample:")
    print(X_test_w[0])

    print("\nFirst test target:")
    print(y_test_w[0])
    model = train_and_save_model(X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w)
    #model = tf.keras.models.load_model(
    #    "carla_mlp_controller.keras"#, 
    #    #custom_objects={"SmoothHuberLoss": SmoothHuberLoss}
    #)
    
    y_pred = model.predict(X_test_w, verbose=0).flatten()

    print("\n" + "=" * 70)
    print("SMALL STEERING - SIGN ACCURACY")
    print("=" * 70)

    abs_y = np.abs(y_test_w)

    for lo, hi in [
        (0.005, 0.015),
        (0.015, 0.025),
        (0.025, 0.040),
        (0.040, 0.070),
        (0.070, 0.120),
    ]:

        mask = (abs_y >= lo) & (abs_y < hi)

        actual = y_test_w[mask]
        pred = y_pred[mask]

        # odstranimo skoraj ničelne predikcije
        sign_mask = np.abs(pred) > 0.001

        actual_sign = np.sign(actual[sign_mask])
        pred_sign = np.sign(pred[sign_mask])

        sign_accuracy = np.mean(actual_sign == pred_sign)

        print(
            f"{lo:.3f}-{hi:.3f}: "
            f"N={np.sum(mask):6d}, "
            f"sign_accuracy={sign_accuracy:.3f}, "
            f"pred_nonzero={np.mean(sign_mask):.3f}"
        )
    print("\n" + "=" * 70)
    print("ACTUAL vs PREDICTED STEERING - CALIBRATION")
    print("=" * 70)

    # Absolutna velikost dejanskega steeringa
    abs_y = np.abs(y_test_w)

    bins = [
        (0.00, 0.01),
        (0.01, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
    ]

    for lo, hi in bins:
        mask = (abs_y >= lo) & (abs_y < hi)

        if np.sum(mask) == 0:
            continue

        actual = y_test_w[mask]
        pred = y_pred[mask]

        print(
            f"|steer| {lo:.2f}-{hi:.2f}: "
            f"N={np.sum(mask):6d}, "
            f"actual_mean={np.mean(actual): .6f}, "
            f"pred_mean={np.mean(pred): .6f}, "
            f"actual_abs={np.mean(np.abs(actual)): .6f}, "
            f"pred_abs={np.mean(np.abs(pred)): .6f}"
        )

    print("\n" + "=" * 70)
    print("ACTUAL vs PREDICTED - SELECTED STEERING VALUES")
    print("=" * 70)

    # Kako blizu je napoved glede na konkretno velikost steeringa
    target_ranges = [
        (0.005, 0.015),
        (0.015, 0.025),
        (0.025, 0.04),
        (0.04, 0.07),
        (0.07, 0.12),
        (0.12, 0.20),
        (0.20, 0.30),
        (0.30, 0.50),
        (0.50, 0.70),
    ]

    for lo, hi in target_ranges:
        mask = (abs_y >= lo) & (abs_y < hi)

        if np.sum(mask) == 0:
            continue

        actual_abs = np.abs(y_test_w[mask])
        pred_abs = np.abs(y_pred[mask])

        print(
            f"{lo:.3f}-{hi:.3f}: "
            f"N={np.sum(mask):6d}, "
            f"actual_abs={np.mean(actual_abs):.5f}, "
            f"pred_abs={np.mean(pred_abs):.5f}, "
            f"ratio={np.mean(pred_abs) / (np.mean(actual_abs) + 1e-8):.3f}"
        )
    abs_y = np.abs(y_test_w)
    abs_err = np.abs(y_pred - y_test_w)

    ranges = [
        (0.00, 0.01),
        (0.01, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.40),
        (0.40, 1.00),
    ]

    print("\n===== ERROR BY STEERING MAGNITUDE =====")

    for lo, hi in ranges:
        mask = (abs_y >= lo) & (abs_y < hi)

        if np.any(mask):
            print(
                f"|steer| {lo:.2f}-{hi:.2f}: "
                f"N={mask.sum():7d}, "
                f"MAE={abs_err[mask].mean():.6f}, "
                f"actual_mean={abs_y[mask].mean():.6f}, "
                f"pred_mean={np.abs(y_pred[mask]).mean():.6f}"
            )


    # ============================================================
    # POSITIVE vs NEGATIVE
    # ============================================================

    print("\n===== POSITIVE vs NEGATIVE =====")

    for name, mask in [
        ("NEGATIVE", y_test_w < 0),
        ("POSITIVE", y_test_w > 0),
    ]:
        if np.any(mask):
            print(
                f"{name}: "
                f"N={mask.sum():7d}, "
                f"MAE={np.mean(np.abs(y_pred[mask] - y_test_w[mask])):.6f}, "
                f"actual_mean={np.mean(np.abs(y_test_w[mask])):.6f}, "
                f"pred_mean={np.mean(np.abs(y_pred[mask])):.6f}"
            )
    evaluate_steering_classes(y_pred=y_pred.flatten(), y_true=y_test_w)
    analyze_model_predictions(model=model, X_test=X_test_w, y_test=y_test_w)
    plot_actual_vs_predicted(model=model, X_test=X_test_w, y_test=y_test_w, max_samples=100000)
    check_steering_consistency(X_test=X_test_w, y_test=y_test_w, model=model, window_size=10, scaler=scaler)