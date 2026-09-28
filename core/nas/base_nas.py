"""Neural Architecture Search (NAS) Engine.

This module defines the Multi-Objective Evolutionary Optimization problem using pymoo.
It searches for optimal neural network architectures balancing test accuracy and TinyML 
hardware constraints (model size, inference latency).
"""

from typing import Any, Callable, Dict, List, Type

import numpy as np
from abc import ABC, abstractmethod
from core.model.base_models import BaseModel
from core.train.train import ModelsTraining
from pymoo.core.problem import ElementwiseProblem
from config import GlobalConfig
from pymoo.core.callback import Callback
import json
import os

# Documentation:
# https://pymoo.org/problems/elementwise.html
# https://pymoo.org/getting_started/part_2.html

class BaseNAS(ElementwiseProblem, ABC):
    """Neural Architecture Search (NAS) problem definition for autonomous driving models.

    Optimizes dynamic network hidden layer structures, activation functions, and layer
    normalization settings using multi-objective evolutionary algorithms.
    """
    
    def __init__(self,  
                 config: GlobalConfig, 
                 xl: List[int], 
                 xu: List[int], 
                 n_obj:int, 
                 n_ieq_constr: int, 
                 model_cls: Type[BaseModel], 
                 trainer_cls: Type[ModelsTraining], 
                 **kwargs
                ) -> None:
        """Initializes the NAS multi-objective problem boundaries and constraints.

        Args:
            config (GlobalConfig): Global framework configuration object.
            xl (List[int]): Lower bounds array of the search space.
            xu (List[int]): Upper bounds array of the search space.
            n_obj (int): Number of objective functions to optimize.
            n_ieq_constr (int): Number of inequality constraints.
            model_cls (Type[BaseModel]): Concrete model class to instantiate during search.
            trainer_cls (Type[ModelsTraining]): Trainer class used for evaluation.
            **kwargs: Additional keyword arguments passed to ElementwiseProblem.

        Raises:
            TypeError: If model_cls is not a subclass of BaseModel or trainer_cls of ModelsTraining.
        """
        if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
            raise TypeError(f"[Core Error] model_cls must be a subclass of BaseModel. Got: {model_cls}")
        
        if not isinstance(trainer_cls, type) or not issubclass(trainer_cls, ModelsTraining):
            raise TypeError(f"[Core Error] trainer_cls must be a subclass of BaseTraining. Got: {trainer_cls}")
        
        self.config: Any = config
        self.decoding_rules: Dict[str, Dict[str, Any]] = {}
        self.model_cls = model_cls
        self.trainer_cls = trainer_cls

        super().__init__(
            n_var=len(xl),
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=np.array(xl),
            xu=np.array(xu),
            vtype=int,
            **kwargs
        )
        print(f"[NAS] Initialized search space with {len(xl)} decision variables.")

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Decodes a flat continuous/integer vector into structured architectural parameters.

        Args:
            x (np.ndarray): Decision variable vector from the optimizer.

        Returns:
            Dict[str, Any]: Mapped dictionary of architectural parameters.
        """
        
        x_rounded = np.round(x).astype(int)
        decoded_params: Dict[str, Any] = {}

        for param_name, rule in self.decoding_rules.items():
            if rule["type"] == "layers":
                start, end = rule["range"]
                decoded_params[param_name] = list(x_rounded[start:end])
            elif rule["type"] == "choice":
                idx = rule["index"]
                choices = rule["choices"]
                choice_idx = x_rounded[idx]
                decoded_params[param_name] = choices[choice_idx] if choice_idx < len(choices) else choices[0]

        return decoded_params

    @abstractmethod
    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Evaluates a candidate architecture.

        Note:
            Subclasses must implement this method to define domain-specific evaluation, 
            training orchestration, and objective/constraint assignment.

        Args:
            x (np.ndarray): Candidate architecture decision variables.
            out (Dict[str, Any]): Dictionary to store evaluation outputs, 
                specifically objectives (``F``) and constraints (``G``).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

# This class is used for generation of json file
class NASHistoryCallback(Callback):
    """Callback for saving all candidate evaluations across all generations."""

    def __init__(self, metric_names: List[str], output_path: str = "core/exported_models/nas_history.json") -> None:
        super().__init__()
        self.output_path: str = output_path
        self.history_records: List[Dict[str, Any]] = []
        self.metric_names: List[str] = metric_names

    def notify(self, algorithm: Any) -> None:
        gen: int = int(algorithm.n_gen)
        pop: Any = algorithm.pop

        for idx, individual in enumerate(pop):
            f_values = list(individual.F) if individual.F is not None else []
            metrics: Dict[str, float] = {}
            for i, val in enumerate(f_values):
                metric_name = (
                    self.metric_names[i] 
                    if i < len(self.metric_names) 
                    else f"objective_{i}"
                )
                metrics[metric_name] = float(val)
            is_pareto = individual in algorithm.opt

            record: Dict[str, Any] = {
                "gen": gen,
                "candidate_idx": idx,
                "is_pareto": is_pareto,
                **metrics  # Dynamically unpacks all domain metrics (e.g., val_loss, tinyml_cost)
            }
            self.history_records.append(record)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history_records, f, indent=2)