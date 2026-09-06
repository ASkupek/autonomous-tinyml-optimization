"""Autonomous Driving Neural Network Architecture.

This module defines a dynamic PyTorch neural network model capable of processing
vehicle telemetry inputs and predicting driving controls (e.g., steering).
Supports dynamic layer configurations and optional layer normalization.
"""

from typing import List

import torch
import torch.nn as nn

from config import GLOBAL_CONFIG, GlobalConfig

#Documentation and blogs:
#https://medium.com/data-scientists-diary/advanced-guide-to-using-nn-modulelist-in-pytorch-da4d49c109fc
#https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
#https://docs.pytorch.org/docs/2.13/index.html
#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html


# TODO(v0.2.0): Rename class from AutonomousDriving to DynamicMLP or GenericMLP 
#               since the dynamic architecture engine is fully dataset-agnostic.
# TODO(v0.2.0): Refactor output activation splitting in forward() to be dynamic based on config 
#               (e.g., config-driven activation maps per output index instead of hardcoded 2:3 and 0:2 slices).
# TODO(v0.2.0): Creation of the model and especially dropout need to be generic and not hardcoded. This should lead to rewrite of _forward method also.
class AutonomousDriving(nn.Module):
    """A PyTorch neural network model for autonomous driving tasks.

    This model is designed to process input features and predict target outputs
    relevant to autonomous driving, such as steering angles, speed, and other
    driving-related metrics.
    """

    def __init__(self, layer_structure: List[int], activation_type: str = "relu", layer_norm: bool = False, config: GlobalConfig = GLOBAL_CONFIG) -> None:
        """Initializes the neural network architecture based on the provided configuration.

        Args:
            layer_structure (List[int]): List of hidden layer dimensions.
            activation_type (str): Type of activation function ('relu', 'tanh', 'sigmoid'). Defaults to "relu".
            layer_norm (bool): Enables Layer Normalization between hidden layers. Defaults to False.
            config (GlobalConfig): Object containing network and simulation parameters. Defaults to GLOBAL_CONFIG.
        """
        #Initialize the PyTorch nn.Module
        super(AutonomousDriving, self).__init__()
        self.config: GlobalConfig = config
        self.layer_norm: bool = layer_norm
        self.activation_function: nn.Module = self._get_activation_function(activation_type=activation_type)

        #Generate model layers based on the input data
        self.input_size: int = len(self.config.pipeline.input_features)
        self.output_size: int = len(self.config.pipeline.target_outputs)

        # Initialize the GRU layer for sequential input processing
        self.hidden_size = int(layer_structure[0])

        # Define the neural network architecture
        # We define module list since our arhitecture will be dynamic based no evolutionary algorithm. 
        self.module_list: nn.ModuleList = nn.ModuleList()

        # Initialize the GRU layer for sequential input processing
        # What is GRU? Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture that is designed to handle sequential data. It is a simpler alternative to the Long Short-Term Memory (LSTM) network, which is another popular RNN variant. GRUs are particularly useful for tasks involving time series data, natural language processing, and other applications where the order of data points matters.
        # In this project, the GRU processes a temporal window of vehicle telemetry
        # (e.g. 10 consecutive time steps) and produces a representation that is then
        # passed through fully connected layers to predict the steering angle.
        # A the moment we are always keeping it at the first layer of the network, but in the future we can also make it dynamic and let the evolutionary algorithm decide if it wants to use it or not.
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )


        self.activation_function = self._get_activation_function(
            activation_type
        )

        # To normalize the outputs of the layers we will use layer normalization. This is optional and can be turned on/off.
        # This is usefull to prevent exploding/vanishing gradients and to speed up the training process.
        # Bad for MCU based models since it adds additional computations and memory usage.
        if self.layer_norm:
            self.norm_list: nn.ModuleList = nn.ModuleList()
        else:
            self.norm_list = None

        self._build_network(layer_structure=layer_structure)

    def _build_network(self, layer_structure: List[int]):
        """Builds or rebuilds the neural network layers dynamically.

        Args:
            layer_structure (list[int], optional): List of hidden layer dimensions. Defaults to None.
            
        Raises:
            ValueError: If `layer_structure` is None or unprovided.
            Exception: Re-raises any execution failure encountered during layer instantiation.
        """
        print(f"[Model] Building architecture with layer structure: {layer_structure}")

        try:
            #Lets build now the structure of the network
            if layer_structure is None:
                raise ValueError("[Model Error] Layer structure must be provided for the neural network architecture.")

            # Clear/reset containers before building (essential for dynamic mutation)
            self.module_list = nn.ModuleList()
            if self.layer_norm:
                self.norm_list = nn.ModuleList()

            #store input size into temp_input_size variable to use it in the loop
            temp_input_size = self.hidden_size

            # We start from 1 since the first layer is GRU and we already define it.
            for hidden_layer in layer_structure[1:]:
                if hidden_layer <= 0:
                    continue # Skip layer: Maybe our genetic algorithm will set to 0 if he want to remove the layer.
                self.module_list.append(nn.Linear(temp_input_size, hidden_layer))

                if self.layer_norm and self.norm_list is not None:
                    self.norm_list.append(nn.LayerNorm(hidden_layer))

                self.module_list.append(nn.Dropout(0.1))
                
                temp_input_size = hidden_layer

            self.module_list.append(nn.Linear(temp_input_size, self.output_size))
            print(
                f"[Model] Built successfully: Input({self.input_size}) -> "
                f"Hidden({len(self.module_list) - 1} layers) -> Output({self.output_size})"
            )

        except Exception as error:
            print(f"[Model Error] Failed to build network layers: {error}")
            raise error

    def build_network(self, layer_structure: List[int]) -> None:
            """Public interface to dynamically rebuild the neural network layers."""
            self._build_network(layer_structure=layer_structure)

    def _get_activation_function(self, activation_type: str) -> nn.Module:
        """Returns the activation function based on the specified type.

        Args:
            activation_type (str): Type of activation function ('relu', 'tanh', 'sigmoid').

        Returns:
            nn.Module: Corresponding PyTorch activation function module.

        Raises:
            ValueError: If an unsupported activation string is provided.
        """
        if activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "tanh":
            return nn.Tanh()
        elif activation_type == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"[Model Error] Unsupported activation type: {activation_type}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor containing driving features.

        Returns:
            torch.Tensor: Output tensor containing predicted target values.

        Raises:
            Exception: Re-raises any failure during execution of the forward pass.

        Example:
                Assume input tensor shape `[50, 10, 4]` (50 samples, 10 time steps,
                4 driving features) and `layer_structure=[93, 34, 55, 67]`.

                Layer construction:
                    * GRU: `nn.GRU(input_size=4, hidden_size=93)`
                    * Layer 0: `nn.Linear(in_features=93, out_features=34)`
                    * Layer 1: `nn.Linear(in_features=34, out_features=55)`
                    * Layer 2: `nn.Linear(in_features=55, out_features=67)`
                    * Output: `nn.Linear(in_features=67, out_features=1)`

                Forward execution flow per batch:

                    1. **Input:**
                    `x` with shape `[50, 10, 4]`

                    Each sample contains 10 consecutive time steps:
                        `[speed, acceleration, lane_offset, heading_error]`

                    2. **GRU:**
                    The complete temporal sequence is passed through the GRU.

                    Input shape:
                        `[50, 10, 4]`

                    GRU output shape:
                        `[50, 10, 93]`

                    The GRU maintains temporal information from previous time steps.

                    3. **Last GRU timestep:**
                    Only the output of the last timestep is used:

                        `gru_output[:, -1, :]`

                    Shape:
                        `[50, 93]`

                    This produces a 93-dimensional representation of the
                    temporal driving history.

                    4. **Fully connected Layer 0:**
                    Linear transformation:

                        `[50, 93] -> [50, 34]`

                    Optional LayerNorm is applied if enabled.

                    Activation function is then applied.

                    5. **Fully connected Layer 1:**
                    Linear transformation:

                        `[50, 34] -> [50, 55]`

                    Optional LayerNorm is applied if enabled.

                    Activation function is then applied.

                    6. **Fully connected Layer 2:**
                    Linear transformation:

                        `[50, 55] -> [50, 67]`

                    Optional LayerNorm is applied if enabled.

                    Activation function is then applied.

                    7. **Output layer:**
                    Linear transformation:

                        `[50, 67] -> [50, 1]`

                    No hidden activation or normalization is applied here.

                    8. **Tanh:**
                    The raw output is passed through `tanh` to constrain the
                    predicted steering value:

                        `[50, 1] -> [50, 1]`

                    Final output contains the predicted steering values.
            """
        try:
            num_hidden = len(self.module_list) - 1
            gru_output, _ = self.gru(x)

            # we are returning only the last output of the GRU since we are interested in the last time step prediction. The GRU is the only layer that is aware of the timesamples.
            x = gru_output[:, -1, :]
            for i in range(0,num_hidden,2):
                # This basically take our nn.Linear and put the x values inside for calculation.
                # Then it overwrites the x with the output of calculation. This is done for all hidden layers.
                x = self.module_list[i](x)
                # If layer normalization is enabled, then apply to the newly calculated x values.
                # This will normalize the values to have mean 0 and variance 1 also in hidden layers..
                
                if self.layer_norm and self.norm_list is not None:
                    x = self.norm_list[i // 2](x)

                # Apply the activation function to introduce non-linearity. This is important for the model to learn complex patterns.
                x = self.activation_function(x)
                x = self.module_list[i + 1](x)
            # We take the last layer and apply it to the x values.
            # This is the output layer and it will give us the final predictions for our target outputs.
            raw_out = self.module_list[-1](x)

            # Since steer is defined in carla between -1 and 1 we decide to add activation function on output layer.
            steer = torch.tanh(raw_out)
          
            return steer
        except Exception as error:
            print(f"[Model Error] Execution failed during forward pass: {error}")
            raise error