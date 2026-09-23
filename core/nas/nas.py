"""Neural Architecture Search (NAS) Engine.

This module defines the Multi-Objective Evolutionary Optimization problem using pymoo.
It searches for optimal neural network architectures balancing test accuracy and TinyML 
hardware constraints (model size, inference latency).
"""

import time
import traceback
from typing import Any, Callable, Dict, List

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import torch
from torch.utils.data import DataLoader
from config import GLOBAL_CONFIG, GlobalConfig
from .models import AutonomousDriving
from .train import ModelsTraining

#For GUI save the candidate data into json
from pymoo.core.callback import Callback
import json
import os

# Documentation:
# https://pymoo.org/problems/elementwise.html
# https://pymoo.org/getting_started/part_2.html


# TODO(v0.2.0): Replace simulated CPU latency benchmarking with actual NXP MCXN947 MCU profiling values
#               or a lookup table based on hardware-in-the-loop (HIL) cycle measurements.
# TODO(v0.2.0): Add activation functions into the config file and start using count from there.
# TODO(v0.2.0): Latency weight is potentional rist. We need to use it smartly
# TODO(v0.2.0): Activation function and normalization for xl and xu need to be synamically assigned.
# TODO(v0.2.0): Calculation of an error need to be updated, at the moment we are giving higher error if the steering angle is bigger. We want to cover also small turns better.
class NAS(ElementwiseProblem):
    """Neural Architecture Search (NAS) problem definition for autonomous driving models.

    Optimizes dynamic network hidden layer structures, activation functions, and layer
    normalization settings using multi-objective evolutionary algorithms.
    """
    
    def __init__(self,  train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader, trainer_factory: Callable[[List[int], str, bool], ModelsTraining], model_factory: Callable[[List[int], str, bool], AutonomousDriving], config: GlobalConfig = GLOBAL_CONFIG) -> None:
        """Initializes the NAS multi-objective problem boundaries and constraints.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            validation_loader (DataLoader): PyTorch DataLoader for validation data.
            test_loader (DataLoader): PyTorch DataLoader for testing data.
            trainer_factory (Callable): Factory function taking (layer_structure, activation_type, layer_norm)
                and returning an initialized ModelsTraining object.
            model_factory (Callable): Factory function taking (layer_structure, activation_type, layer_norm)
                and returning an initialized AutonomousDriving model instance.
            config (GlobalConfig): Configuration object containing search space hyperparameters.
        """
        
        self.config: GlobalConfig = config
        self.num_of_epochs: int = self.config.ai.num_of_epochs
        self.num_of_hidden_layers: int = self.config.ai.num_of_hidden_layers

        print("[NAS] Pre-loading dataset and DataLoaders...")
        
        self.train_loader: DataLoader = train_loader
        self.validation_loader: DataLoader = validation_loader
        self.test_loader: DataLoader = test_loader

        self.trainer_factory: Callable[[List[int], str, bool], ModelsTraining] = trainer_factory
        self.model_factory: Callable[[List[int], str, bool], AutonomousDriving] = model_factory

        print("[NAS] DataLoaders successfully cached for NAS search.")

        xl: List[int] = []
        xu: List[int] = []

        # For each layer we add upper and lower limits for the number of neurons in that layer. The lower limit is the minimum hidden layer size, and the upper limit is the maximum hidden layer size.
        for _ in range(self.num_of_hidden_layers):
            xl.append(self.config.ai.min_hidden_layer_size)
            xu.append(self.config.ai.max_hidden_layer_size)

        # The next variable is the activation function type, Min and Max values. At the moment we are using only 1. Relu
        xl.append(0)
        xu.append(0)
        
        # The last variable is the layer normalization type
        xl.append(0)
        # We set also upper limit to 0, to avoid normalization
        xu.append(0)
        
        # n_obj is number of objectives in our case 2: accuracy and tinyML optimization (time and space)
        # n_ieq_constr is number is hard constraints, in our case 2:
            # 1. The number of neurons must be greater than 0.
            # 2. The size of tinyML must be less than x kB
        super().__init__(
            n_var=len(xl), 
            n_obj=2, 
            n_ieq_constr=2, 
            xl=np.array(xl), 
            xu=np.array(xu), 
            vtype=int
        )
        print(f"[NAS] Initialized search space with {len(xl)} decision variables.")

    def get_model_size_bytes(self, model: torch.nn.Module) -> int:
        """Calculates total trainable parameter memory footprint in bytes (float32).

        Args:
            model (torch.nn.Module): Target PyTorch network model.

        Returns:
            int: Size of model weights in bytes.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4

    def get_activation_function(self, activation_type: int) -> str:
        """Maps integer index representations to string activation names.

        Args:
            activation_type (int): Integer index (0: 'relu', 1: 'tanh', 2: 'sigmoid').

        Returns:
            str: Name of activation function.

        Raises:
            ValueError: If an invalid activation index is provided.
        """
        if activation_type == 0:
            return "relu"
        elif activation_type == 1:
            return "tanh"
        elif activation_type == 2:
            return "sigmoid"
        else:
            raise ValueError(f"[NAS Error] Invalid activation index: {activation_type}.")
    
    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs:Any) -> None:
        """Evaluates the performance of the neural network model based on the given architecture parameters.

        Args:
            x (np.ndarray): Array containing the architecture parameters (hidden layer sizes, activation type, layer normalization).
            out (dict): Dictionary to store the evaluation results.
        """
        
        x_rounded = np.round(x).astype(int)

        # Define the architecture parameters based on the rounded input values
        # layer_structure is a list of integers representing the number of neurons in each hidden layer, derived from the first num_of_hidden_layers elements of x_rounded.
        layer_structure = [x_rounded[i] for i in range(self.num_of_hidden_layers)]

        # activation_type is a string representing the activation function to be used in the neural network, derived from the second-to-last element of x_rounded.
        activation_type = self.get_activation_function(activation_type=x_rounded[self.num_of_hidden_layers])

        # layer_norm is a boolean representing whether to use layer normalization in the neural network, derived from the last element of x_rounded.
        layer_norm = bool(x_rounded[-1])

        print(f"[NAS] Evaluating Candidate: Layers={layer_structure}, Act={activation_type}, Norm={layer_norm}")

        # For first penalty
        structure_check = [l for l in layer_structure if l > 0]

        # First penalty: If the number of neurons is 0, we will give a penalty to the model. This is to prevent the model from having no neurons in the hidden layers.
        # Negative value means that the model is valid
        g1 = -1.0 if len(structure_check) > 0 else 1.0

        # Now we will make a model to get the size of it in kB.
        # Only if structure_check is bigger that 0
        if len(structure_check) > 0:
            try:
                model = self.model_factory(
                    layer_structure=layer_structure,
                    activation_type=activation_type,
                    layer_norm=layer_norm
                )
                #model_size_kb = self.get_model_size_bytes(model) / 1024.0
                model_size_kb = (sum(p.numel() for p in model.parameters() if p.requires_grad) * 4) / 1024.0

                # Second penalty: If the size of the model is bigger than the maximum size defined in the config, we will give a penalty to the model.
                g2 = model_size_kb - self.config.ai.max_model_size_kb

                # Now we measure latency of the model with random input data.
                # At the moment we symulate latency but in future we can use MCU symulator -> this will give us more valid results.

                dummy_input = torch.randn(
                    1,
                    self.config.pipeline.window_size,
                    len(self.config.pipeline.input_features)
                )

                with torch.no_grad():
                    # Let's start the timer
                    start = time.perf_counter()

                    for _ in range(30):
                        _ = model(dummy_input)
                    # Let's stop the timer devide by 30 to get the average and multiply by 1000 to get the time in ms.
                    inference_ms = ((time.perf_counter() - start) / 30.0) * 1000
                # Calculate the combined cost between model size and latency
                # This is our tinyML cost
                tiny_ml_cost = model_size_kb + (inference_ms * self.config.ai.latency_weight)
            except Exception as error:
                print(f"[NAS Error] Failed to evaluate model candidate structure: {error}")
                g2 = 1.0
                tiny_ml_cost = 999.0
        else:
            # If generated structure is equal to 0, we give hard penalties
            g2 = 1.0
            tiny_ml_cost = 999.0

        out["G"] = np.array([g1, g2], dtype=np.float64)

        # Now we move to second penalty
        # This is in our case the how accurate the model is
        if g1 <= 0 and g2 <= 0:
            try:
                trainer = self.trainer_factory(
                    layer_structure=layer_structure,
                    activation_type=activation_type,
                    layer_norm=layer_norm,
                    model=model
                )
                trainer.run_training() 
                
                model.eval()
                preds_list, targets_list = [], []
                with torch.no_grad():
                    # We iterate over the validation DataLoader to obtain model predictions and corresponding ground-truth targets for each batch.
                    for inputs, targets in self.validation_loader:
                        outputs = model(inputs)
                        preds_list.extend(outputs.flatten().cpu().numpy())
                        targets_list.extend(targets.flatten().cpu().numpy())
                
                y_pred = np.array(preds_list)
                y_true = np.array(targets_list)
                
                # We give an error based on the absolute error between the predicted and true values, weighted by the magnitude of the steering angle. This prioritizes accuracy on sharp turns.
                abs_true = np.abs(y_true)
                weights = np.ones_like(abs_true)
                
                # We assign weights based on the absolute value of the true steering angle. The weights are set to prioritize sharper turns, with higher weights for larger steering angles.
                weights[abs_true > 0.02] = 2.0  
                weights[abs_true > 0.1]  = 5.0  
                weights[abs_true > 0.3]  = 10.0
                # We calculate the weighted mean absolute error (MAE) as the validation metric, which serves as the first objective for optimization.
                absolute_errors = np.abs(y_pred - y_true)
                weighted_val_metric = np.average(absolute_errors, weights=weights)

                print(f"[NAS] Candidate Validation Metric (Weighted MAE): {weighted_val_metric:.6f}, TinyML Cost: {tiny_ml_cost:.3f} kB+ms")
                # Minimize two goals:
                # Goal 1: loss of the model
                # Goal 2: TinyML cost
                out["F"] = np.array([weighted_val_metric, tiny_ml_cost], dtype=np.float64)
            except Exception as error:
                print(f"[NAS Error] Training pipeline failed during optimization: {error}")
                traceback.print_exc()
                out["F"] = np.array([999.0, tiny_ml_cost], dtype=np.float64)
        else:
            # If we already fail on g1 and g2, put max penalties inside
            out["F"] = np.array([999.0, tiny_ml_cost], dtype=np.float64)

# This class is used for generation of json file
class NASHistoryCallback(Callback):
    """Callback for saving all candidate evaluations across all generations."""

    def __init__(self, output_path: str = "ml_pipeline/exported_models/nas_history.json") -> None:
        super().__init__()
        self.output_path: str = output_path
        self.history_records: List[Dict[str, Any]] = []

    def notify(self, algorithm: Any) -> None:
        gen: int = int(algorithm.n_gen)
        pop: Any = algorithm.pop

        for idx, individual in enumerate(pop):
            val_loss: float = float(individual.F[0]) if individual.F is not None else 999.0
            tinyml_cost: float = float(individual.F[1]) if individual.F is not None else 999.0

            is_pareto = individual in algorithm.opt

            record: Dict[str, Any] = {
                "gen": gen,
                "candidate_idx": idx,
                "val_loss": val_loss,
                "tinyml_cost": tinyml_cost,
                "is_pareto": is_pareto,
            }
            self.history_records.append(record)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history_records, f, indent=2)


        