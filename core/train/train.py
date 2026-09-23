"""Model Training and Evaluation Pipeline.

This module handles the training, validation, and evaluation loops for the
Autonomous Driving PyTorch neural network model.
"""

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import GLOBAL_CONFIG, GlobalConfig
from .models import AutonomousDriving
from typing import List, Optional



# TODO(v0.2.0): Remove or fully deprecate _compute_weighted_loss if Smart Downsampling 
#               completely resolves the steering class imbalance.
#TODO(v0.2.0): Implement early stop
#

class ModelsTraining:
    """Handles the training and evaluation of the autonomous driving neural network model."""

    def __init__(self, train_loader: DataLoader, validation_loader: DataLoader, layer_structure: List[int], activation_type: str = "relu", layer_norm: bool = False, config: GlobalConfig = GLOBAL_CONFIG, model: Optional[nn.Module] = None) -> None:
        """Initializes the training manager with datasets, network architecture, and execution device.

        Args:
            train_loader (DataLoader): Training dataset used to optimize the model parameters.
            validation_loader (DataLoader): Validation dataset used to evaluate the model during training.
            layer_structure (List[int]): Architecture configuration for hidden layers.
            activation_type (str): Type of activation function ('relu', 'tanh', 'sigmoid'). Defaults to "relu".
            layer_norm (bool): Whether to enable Layer Normalization. Defaults to False.
            config (GlobalConfig): Global configuration object with AI hyperparameters. Defaults to GLOBAL_CONFIG.
            model (Optional[nn.Module]): Pre-instantiated PyTorch model. If None, instantiates AutonomousDriving.
        
        """
        self.config: GlobalConfig = config
        self.layer_structure: List[int] = layer_structure
        self.activation_type: str = activation_type
        self.layer_norm: bool = layer_norm
        self.num_of_epochs: int = self.config.ai.num_of_epochs
        # Let's check if there is GPU, otherwise we will run on CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the neural network model with the specified architecture and settings.
        if model is not None:
            self.model: nn.Module = model.to(self.device)
        else:
            self.model: nn.Module = AutonomousDriving( 
                layer_structure=self.layer_structure, 
                activation_type=self.activation_type, 
                layer_norm=self.layer_norm,
                config=self.config
            ).to(self.device)
        
        
        self.train_loader = train_loader 
        self.validation_loader = validation_loader

        # Define the loss function and optimizer for training the model.
        self.loss = nn.MSELoss()
        lr: float = self.config.ai.learning_rate
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr = lr)
        self.epoch_validation_loss: float = 0.0

    def _compute_weighted_loss(self, outputs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates a custom weighted Mean Squared Error (MSE) loss.

        Samples with larger ground-truth steering angles receive higher weights.
        This prioritizes accurate predictions during turns and recovery maneuvers
        over straight-line driving.


        Args:
            outputs (torch.Tensor): Predicted model outputs of shape (batch_size, num_outputs).
            y_true (torch.Tensor): Ground-truth target values of shape (batch_size, num_outputs).

        Returns:
            torch.Tensor: Scalar tensor representing the mean weighted loss for the batch.
        """
        y_true_reshaped = y_true.view_as(outputs)

        # Squared error between predicted and true values
        error = torch.square(outputs - y_true_reshaped)

        # absolute value of the true steering angle to determine the weight for each sample
        abs_y = torch.abs(y_true_reshaped)

        
        # Steering samples are weighted according to their magnitude.
        # The weight increases linearly from 1.0 at |steer| = 0.0
        # to 3.0 at |steer| = 0.1 and is capped at 3.0.
        #
        # |steer| = 0.00 -> weight = 1.0
        # |steer| = 0.05 -> weight = 2.0
        # |steer| >= 0.10 -> weight = 3.0
        weights = 1.0 + 2.0 * torch.clamp(
            abs_y / 0.1,
            min=0.0,
            max=1.0
        )
        # Return weighted MSE
        return torch.mean(weights * error)

    def run_training(self) -> nn.Module:
        """Runs the training and evaluation loop for the neural network model.
        
        Returns:
            nn.Module: The trained neural network model.
            
        Raises:
            Exception: Re-raises any error encountered during training or evaluation iterations.    
        """
        print(f"[Trainer] Starting training process for {self.num_of_epochs} epochs...")
        try:
            for epoch in range(self.num_of_epochs):
                # Put the model into training mode
                self.model.train()
                train_loss = 0.0
                for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                    # Move the data to the appropriate device (GPU or CPU)
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    # Zero the gradients from the previous step. To not accumulate the losses.
                    self.optimizer.zero_grad()

                    # Forward pass: compute predicted outputs by passing inputs to the model
                    # This one calls forward method of the model and calculate the output of the model based on the input features.
                    outputs = self.model(X_batch)

                    # Compute the loss between predicted and actual values
                    if outputs.ndim != y_batch.ndim:
                        y_batch = y_batch.view(-1, 1)
                    #loss_value = self.loss(outputs, y_batch)
                    loss_value = self._compute_weighted_loss(outputs, y_batch)
                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss_value.backward()

                    # Perform a single optimization step (parameter update)
                    self.optimizer.step()

                    # Accumulate the training loss for reporting
                    train_loss += loss_value.item()
                epoch_train_loss = train_loss / len(self.train_loader)

                # Move model to evaluation mode for validation
                self.model.eval()
                validation_loss = 0.0
                # Evaluate the model on the validation dataset without computing gradients.
                # Gradients are used during training to update the model parameters, but during evaluation, we only want to compute the loss and predictions.
                with torch.no_grad():
                    for X_validation, y_validation in self.validation_loader:
                        # Move the validation data to the appropriate device
                        X_validation, y_validation = X_validation.to(self.device), y_validation.to(self.device)
                        
                        # Predict the outputs for the validation data using the trained model
                        validation_outputs = self.model(X_validation)
                        
                        # Compute the loss between predicted and actual values for the validation data
                        if validation_outputs.ndim != y_validation.ndim:
                            y_validation = y_validation.view(-1, 1)
                        #validation_loss_value = self.loss(validation_outputs, y_validation)
                        validation_loss_value = self._compute_weighted_loss(validation_outputs, y_validation)
                        validation_loss += validation_loss_value.item()

                self.epoch_validation_loss = validation_loss / len(self.validation_loader)
                print(f" [Trainer] -> Epoch [{epoch+1:02d}/{self.num_of_epochs}] | Average validation MSE: {self.epoch_validation_loss:.6f}") 
                print(f" [Trainer] -> Epoch [{epoch+1:02d}/{self.num_of_epochs}] | Average train MSE: {epoch_train_loss:.6f}")
                print(json.dumps({"trial": epoch+1, "validation loss": self.epoch_validation_loss}), flush=True)
            print("[Trainer] Training completed successfully.")
            return self.model
        except Exception as error:
                print(f"[Trainer Error] Training loop failed on epoch {epoch + 1}: {error}")
                raise error