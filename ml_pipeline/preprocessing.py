"""Data Preprocessing Engine.

This module loads raw telemetry CSV datasets, filters out physical anomalies/outliers,
applies feature scaling, persists scaling artifacts, and converts the data into
PyTorch DataLoaders ready for neural network training.
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import GLOBAL_CONFIG, GlobalConfig

#Documentation:
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#https://docs.pytorch.org/docs/2.13/data.html#torch.utils.data.DataLoader


# TODO(v0.2.0): Refactor smart downsampling logic into a standalone dataset cleaning script 
#               or dedicated module to separate raw I/O transformation from PyTorch DataLoader assembly.
# TODO(v0.2.0): Loading of the downsample dataset is at the moment in one big method. Move this code to separate method and put configs into config file.
# TODO(v0.2.0): Move data_preprocessing_view into helper function since it is not needed anymore.
# TODO(v0.2.0): Check comment here: groups = np.array_split(unique_vehicles, 4)
# TODO(v0.2.0): def save_processed_datasets() need to be removed or moved to separate file since it is not needed anymore. It is only used for debugging and to check the data.

class PyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for CARLA driving telemetry data.
    Converts preprocessed NumPy arrays into PyTorch float32 tensors for efficient
    batching via DataLoader.
    """

    def __init__(self, X:np.ndarray, y: np.ndarray) -> None:
        """Initializes the dataset with features and targets.

        Args:
            X (np.ndarray): Scaled input features.
            y (np.ndarray): Target driving controls.
        """
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(y, dtype=torch.float32)

    # Important for DataLoader to know how to batch the data
    def __len__(self) -> int:
        """Returns the total number of samples. Required by DataLoader.
        Returns:
            int: Number of available sample rows.
        """
        return len(self.X)
    
    # Important for pytorch
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a single sample tuple (features, targets) at the given index.
        Args:
            idx (int): Row index to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing feature tensor (X) and target tensor (y).
        """
        return self.X[idx], self.y[idx]

class DataPreprocessing:
    """Handles end-to-end data cleansing, scaling, and batching operations."""

    def __init__(self, config: GlobalConfig = GLOBAL_CONFIG) -> None:
        """Initializes the preprocessing pipeline with configuration settings.

        Args:
            config (GlobalConfig): Object containing network, topic, and storage configurations.
                Defaults to GLOBAL_CONFIG.
        """
        self.config: GlobalConfig = config
        self.input_features: list[str] = self.config.pipeline.input_features
        self.target_outputs: list[str] = self.config.pipeline.target_outputs

    def read_and_preprocess_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Executes the full pipeline: load, clean, split, scale, and batch data.

        Returns:
            Tuple[DataLoader, DataLoader,DataLoader]: Shuffled train_loader and sequential test_loader.
            
        Raises:
            FileNotFoundError: If the source CSV dataset file does not exist on disk.
            ValueError: If dataset becomes empty after applying boundary constraints.
            Exception: Re-raises any execution failure during the pipeline execution.
        """
        csv_path = self.config.pipeline.csv_path
        print(f"[Data Preprocessing] Loading dataset from '{csv_path}'...")

        try:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Dataset file does not exist at: {csv_path}")

            if not os.path.isfile(self.config.pipeline.path_to_downsampled_dataset):
                dataset = pd.read_csv(csv_path)
                dataset_balanced = (
                        dataset.sort_values(by=["vehicle_id", "timestamp"])
                          .reset_index(drop=True)
                    )
                dataset_balanced.to_csv(self.config.pipeline.path_to_downsampled_dataset, index=False)

            else:
                dataset_balanced = pd.read_csv(self.config.pipeline.path_to_downsampled_dataset)

            dataset = dataset_balanced
            initial_count = len(dataset)
            print(f"[Data Preprocessing] Raw records loaded: {initial_count}")

            # Clean missing values
            dataset_clean = dataset.dropna()

            bounds = getattr(self.config.pipeline, "feature_bounds", {})
            for col, (min_val, max_val) in bounds.items():
                if col in dataset_clean.columns:
                    if min_val is not None:
                        dataset_clean = dataset_clean[dataset_clean[col] >= min_val]
                    if max_val is not None:
                        dataset_clean = dataset_clean[dataset_clean[col] <= max_val]

            cleaned_count = len(dataset_clean)
            dataset = dataset_clean
            print(f"[Data Preprocessing] Valid records remaining after filtering: {cleaned_count} "
                  f"({initial_count - cleaned_count} removed)")
            
            if cleaned_count == 0:
                raise ValueError("Dataset is empty after applying boundary constraints and dropping NaNs.")

            # We sort dataset byy vehicle_id and timestamp to ensure that the data is in chronological order for each vehicle.
            unique_vehicles = sorted(dataset['vehicle_id'].unique())
            # We split the unique vehicle IDs into 4 groups for training, validation, and testing. -> We know that this need to be updated since 4 is num of cities. We need to make this dynamic based on the number of cities in the dataset.
            groups = np.array_split(unique_vehicles, 4)

            # For each dataset split, we select the rows corresponding to the vehicle IDs in the respective groups. This ensures that all data from a single vehicle is contained within one dataset split, preventing data leakage.
            train_dataset = dataset[dataset['vehicle_id'].isin(np.concatenate([groups[0], groups[1]]))]
            validation_dataset = dataset[dataset['vehicle_id'].isin(groups[2])]
            test_dataset = dataset[dataset['vehicle_id'].isin(groups[3])]

            # Save scaler for later to execute on real life testing
            scaler_path = getattr(self.config.pipeline, "scaler_path", self.config.pipeline.scaler_path)
            if(os.path.exists(scaler_path)):
                scaler = joblib.load(scaler_path)
                print(f"[Data Preprocessing] Loaded existing scaler from '{scaler_path}'.")
            else:
                scaler = StandardScaler()
                scaler.fit(train_dataset[self.config.pipeline.input_features])
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(scaler, scaler_path)
                print(f"[Data Preprocessing] StandardScaler successfully saved to '{scaler_path}'.")

            train_dataset[self.input_features] = scaler.transform(train_dataset[self.input_features])
            validation_dataset[self.input_features] = scaler.transform(validation_dataset[self.input_features])
            test_dataset[self.input_features] = scaler.transform(test_dataset[self.input_features])

            # We set the window size for creating time-series input samples based on the configuration.
            window_size = self.config.pipeline.window_size

            # We define a helper function to create sliding windows of input features and corresponding target outputs for each vehicle. This is essential for time-series prediction tasks, where the model needs to learn from sequences of past observations.
            def build_windows(df):
                X_list, y_list = [], []
                #  We group the dataset by vehicle_id to ensure that the sliding windows are created within the context of each individual vehicle's telemetry data. This prevents mixing data from different vehicles, which could introduce noise and reduce model performance.
                for vehicle_id, group in df.groupby('vehicle_id'):
                    # We sort each vehicle's data by timestamp to maintain the chronological order of observations. This is crucial for time-series modeling, as the model needs to learn from past states to predict future states.
                    group_sorted = group.sort_values(by="timestamp")
                    features = group_sorted[self.input_features].to_numpy()
                    targets = group_sorted[self.target_outputs].to_numpy().ravel()
                    
                    for i in range(window_size - 1, len(group_sorted)):
                        window = features[i - window_size + 1: i + 1]
                        X_list.append(window)
                        y_list.append(targets[i])
                        
                return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
            
            X_train_w, y_train_w = build_windows(train_dataset)
            X_val_w, y_val_w = build_windows(validation_dataset)
            X_test_w, y_test_w = build_windows(test_dataset)

            # This is again a helper function that in the end saves the processed datasets into a folder called processed_dataset. This is useful for later use and to avoid repeating the preprocessing steps.
            # At the moment it is uded for debugging
            def save_processed_datasets(
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
            ) -> None:

                dataset_path = "processed_dataset"

                os.makedirs(dataset_path, exist_ok=True)

                np.savez(
                    os.path.join(dataset_path, "train.npz"),
                    X=X_train.astype(np.float32),
                    y=y_train.astype(np.float32),
                )

                np.savez(
                    os.path.join(dataset_path, "validation.npz"),
                    X=X_val.astype(np.float32),
                    y=y_val.astype(np.float32),
                )

                np.savez(
                    os.path.join(dataset_path, "test.npz"),
                    X=X_test.astype(np.float32),
                    y=y_test.astype(np.float32),
                )
            save_processed_datasets(
                X_train=X_train_w,
                y_train=y_train_w,
                X_val=X_val_w,
                y_val=y_val_w,
                X_test=X_test_w,
                y_test=y_test_w,
            )

            # We create PyTorch Dataset objects for training, validation, and testing. These datasets wrap the preprocessed NumPy arrays and convert them into PyTorch tensors, enabling efficient batching and shuffling during model training.
            train_dataset = PyTorchDataset(X=X_train_w, y=y_train_w)
            validation_dataset = PyTorchDataset(X=X_val_w, y=y_val_w)
            test_dataset = PyTorchDataset(X=X_test_w, y=y_test_w)

            # Saving for execute on MCU, so that we don't need to repeat separation and scalling of the values
            self.save_test_data(X_test_scaled=X_test_w, y_test=y_test_w)

            batch_size = self.config.ai.batch_size
            # Put shuffle to false to prevent strange break speed training
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            print(f"[Data Preprocessing] DataLoaders created successfully (Batch size: {batch_size}).")

            return train_loader, validation_loader, test_loader
        
        except Exception as error:
            print(f"[Data Preprocessing Error] Failed during processing pipeline: {error}")
            raise error

    def save_test_data(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> None:
        """Saves the preprocessed test dataset for later model evaluation.

        The test inputs are stored in their scaled form, while the target outputs
        remain in their original numerical representation. The saved dataset can
        later be loaded by ExecuteOnMCU without repeating the train/test split
        or preprocessing pipeline.

        Args:
            X_test_scaled (np.ndarray): Scaled test input features.
            y_test (np.ndarray): Test target outputs.

        Raises:
            ValueError: If test data is missing.
            Exception: If the test dataset cannot be saved to disk.
        """
        try:
            if X_test_scaled is None or y_test is None:
                print(f"[Data Preprocessing]Test data cannot be None.")
                raise ValueError("Test data cannot be None.")

            test_data_path = self.config.pipeline.export_test_data

            directory = os.path.dirname(test_data_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            np.savez(
                test_data_path,
                X_test_scaled=X_test_scaled.astype(np.float32),
                y_test=y_test.astype(np.float32),
            )

            print(f"[Data Preprocessing] Test dataset successfully saved "f"to '{test_data_path}'.")

        except Exception as error:
            print(f"[Data Preprocessing Error] Failed to save test dataset: {error}")

    def data_preprocessing_view(self) ->None:
        """
        Method used for data check during the development of the NN
        """
        csv_path = self.config.pipeline.csv_path
        try:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Dataset file does not exist at: {csv_path}")

            dataset = pd.read_csv(csv_path)

            features = [f for f in self.config.pipeline.input_features if f in dataset.columns]
            stats_X = dataset[features].agg(['min', 'max', 'mean', 'std']).T
            stats_X['abs_max'] = dataset[features].abs().max()

            targets = [t for t in self.config.pipeline.target_outputs if t in dataset.columns]
            stats_y = dataset[targets].agg(['min', 'max', 'mean', 'std']).T
          
        except Exception as error:
            print(f"[Data Preprocessing Error] Dataset preview failed: {error}")