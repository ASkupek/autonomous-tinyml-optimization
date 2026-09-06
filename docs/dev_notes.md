# DEVELOPMENT NOTES
> **Developer Note:** This document serves as an internal, rapid-iteration technical log for the `v0.1.0-alpha` architecture. It prioritizes system context, edge-case tracking, and ongoing code documentation over formal prose. Polished documentation will be updated for the official stable release.
>>> # TODO:
> ## Tehnical debt for each class is mentioned at the .py files and here at the beginning of the class description
> ## We still need to update the main and dashboard
> ## Add tests
> ## Prepare req file
> ## Test complete flow: get data, build networks, test on close and open loop!
> ## Add into documentation how we ported into MCU
> ## Code cleanup: check if all comments make sense and clean the code where needed
> ## Main code explanation is missing here
> ## Comments need to be checked to see if order is correct on the input parameters
> ## This document will be updated
> ## AutonomousDriving -> update the notes to write more regarding the GRU!
---
## Where to:
* [1.0 Project overview](#10-project-overview)
* [2.0 Modules](#20-modules)
* [3.0 Configuration](#30-configuration)
* [4.0 Build Dataset](#40-build-dataset)
    * [4.1 Class: CarlaDataStreaming](#41-class-carladatastreaming)
        * [4.1.0 Class Methods](#410-class-methods)
        * [4.1.1 Constructor: `__init__()`](#411-constructor-__init__)
        * [4.1.2 Class method: `connect()`](#412-class-method-connect)
        * [4.1.3 Class method: `spawn_vehicle()`](#413-class-method-spawn_vehicle)
        * [4.1.4 Class method: `start_streaming()`](#414-class-method-start_streaming)
        * [4.1.5 Class method: `on_delivery()`](#415-class-method-on_delivery)
        * [4.1.6 Class method: `_on_obstacle_detected()`](#416-class-method-_on_obstacle_detected)
        * [4.1.7 Class method: `get_lane_data()`](#417-class-method-get_lane_data)
        * [4.1.8 Class method: `get_vehicle_data()`](#418-class-method-get_vehicle_data)
        * [4.1.9 Class method: `set_carla_data()`](#419-class-method-set_carla_data)
        * [4.1.10 Class method: `cleanup()`](#4110-class-method-cleanup)
    * [4.2 Class: TelemetryConsumer](#42-class-telemetryconsumer)
        * [4.2.0 Class Methods](#420-class-methods)
        * [4.2.1 Constructor: `__init__()`](#421-constructor-__init__)
        * [4.2.2 Class method: `kafka_connection()`](#422-class-method-kafka_connection)
        * [4.2.3 Class method: `kafka_start_listening()`](#423-class-method-kafka_start_listening)
        * [4.2.4 Class method: `flush_buffer()`](#424-class-method-flush_buffer)
        * [4.2.5 Class method: `stop_listening()`](#425-class-method-stop_listening)
* [5.0 TinyML models generation](#50-tinyml-models-generation)
    * [5.1 Class: NAS](#51-class-nas)
        * [5.1.0 Class methods](#510-class-methods)
        * [5.1.1 Constructor: `__init__()`](#511-constructor-__init__)
        * [5.1.2 Class method: `get_model_size_bytes()`](#512-class-method-get_model_size_bytes)
        * [5.1.3 Class method: `get_activation_function`](#513-class-method-get_activation_function)
        * [5.1.4 Class method: `_evaluate()`](#514-class-method-_evaluate)
    * [5.2 Class: DataPreprocessing](#52-class-datapreprocessing)
        * [5.2.0 Class methods](#520-class-methods)
        * [5.2.1 Class constructor: `__init__()`](#521-constructor-__init__)
        * [5.2.2 Class method: `read_and_preprocess_dataset()`](#522-class-method-read_and_preprocess_dataset)
        * [5.2.3 Class method: `save_test_data()`](#523-class-method-save_test_data)
        * [5.2.4 Class method: `data_preprocessing_view()`](#524-class-method-data_preprocessing_view)
    * [5.3 Class: AutonomousDriving](#53-class-autonomousdriving)
        * [5.3.0 Class methods](#530-class-methods)
        * [5.3.1 Class constructor: `__init__()`](#531-constructor-__init__)
        * [5.3.2 Class method: `build_network()`](#532-class-method-build_network)
        * [5.3.3 Class method: `_get_activation_function()`](#533-class-method-_get_activation_function)
        * [5.3.4 Class method: `forward()`](#534-class-method-forward)
    * [5.4 Class: ModelsTraining](#54-class-modelstraining)
        * [5.4.0 Class methods](#540-class-methods)
        * [5.4.1 Class constructor: `__init__()`](#541-constructor-__init__)
        * [5.4.2 Class method: `_compute_weighted_loss()`](#542-class-method-_compute_weighted_loss)
        * [5.4.3 Class method: `run_training()`](#543-class-method-run_training)
    * [5.5 Class: ExportEngine](#55-class-exportengine)
        * [5.5.0 Class methods](#550-class-methods)
        * [5.5.1 Class constructor: `__init__()`](#551-constructor-__init__)
        * [5.5.2 Class method: `verify_models()`](#552-class-method-verify_models)
        * [5.5.3 Class method: `representative_data_gen()`](#553-class-method-representative_data_gen)
        * [5.5.4 Class method: `copy_weights()`](#554-class-method-copy_weights)
        * [5.5.5 Class method: `export_tflite()`](#555-class-method-export_tflite)
        * [5.5.6 Class method: `export_c_header()`](#556-class-method-export_c_header)
        * [5.5.7 Class method: `export_top_models()`](#557-class-method-export_top_models)
* [6.0 Testing](#60-testing)
     [`Class methods`](#61-class-methods)
    * [`__init__()`](#62-constructor-__init__)
    * [`close_loop_testing_CARLA()`](#63-constructor-close_loop_testing_carla)
    * [`_send_to_mcu()`](#64-constructor-_send_to_mcu)
    * [`test_on_dataset()`](#65-constructor-test_on_dataset)
    * [`load_dataset()`](#66-constructor-load_dataset)
* [7.0 Code on the MCU](#70-code-on-the-mcu)
* [8.0 Helper functions](#80-helper-functions)
* [9.0 Problems and Solutions](#90-problems-and-solutions)
* [10.0 Tests](#100-tests)
* [11.0 GUI](#110-gui)
---
## 1.0 Project overview
IN this part of the document we will first separate program by modules, then each module will be described how to execute it
and notes for important function and code inside the functions will be described.

---
## 2.0 Modules
1. **Configuration**
2. **Build dataset**
3. **TinyML models generation**
4. **Test the NN**
    a. Close loop testing
    b. Test on test data

---
## 3.0 Configuration

> ### TODO(v0.2.0): Create separated testing class and move needed flags there and add new ones

Configrations are stored inside config.py. They are separated by the data classes and each class represent some part that is important for execution of the program.
At the moment we have following classes:
* Carla Config
* Kafka Config
* Pipeline Config
* AI Config
* Port Configuration
* Global config

Check the comments in the code to get the idea for what each variable is used for.
Reason for this file is to have one source of truths. User can modify the variables there and not in actual code. This helps us in the adding or changing the variables so we don't need to search in the code, but we can modify only here and complete code is changed correctly.

NOTE: For testing you need to change num of cars to 1 and testing_mcu to True

## 4.0 Build dataset

### 4.1 Class: CarlaDataStreaming
This class is located in carla_streaming.py.


> ### TODO(v0.2.0): Combine break + throttle    
> ### TODO(v0.2.0): User can select town for testing 
> ### TODO(v0.2.0): Getting data in a loop is causing below code killing the loop (method: start_streaming) -> At the moment we comment it out and user need to call it
>         finally:
            self.cleanup()
### 4.1.0 Class Methods: 
* [`__init__()`](#411-constructor-__init__)
* [`connect()`](#412-class-method-connect)
* [`spawn_vehicle()`](#413-class-method-spawn_vehicle)
* [`start_streaming()`](#414-class-method-start_streaming)
* [`on_delivery()`](#415-class-method-on_delivery)
* [`_on_obstacle_detected()`](#416-class-method-_on_obstacle_detected)
* [`get_lane_data()`](#417-class-method-get_lane_data)
* [`get_vehicle_data()`](#418-class-method-get_vehicle_data)
* [`set_carla_data()`](#419-class-method-set_carla_data)
* [`cleanup()`](#420-class-method-cleanup)

### 4.1.1 Constructor: `__init__()` 
Constructor used to initialize the variables needed for code execution.

### 4.1.2 Class method: `connect()`
Establish the connection to all needed parts. Here it is important to understand that we have two possibilities:
* Getting a data for dataset
* Close loop testing
In first one we establish the connection to Carla and Kafka, since Kafka is taking the data and taking care no data is getting lost.
In second one we need only Carla so that is the reasong we have an if statement there,

### 4.1.3 Class method: `spawn_vehicle()` 
Method used for spawning the vehicles.
This method is responsible to spawn vehicles into the Carla map. It will generate the vehicles and sensors that are used. After that it is attaching sensors to the car.
Since same code is used for generating data and testing on close loop we needed to make sure that autopilot is only enabled on gethering the data and not on testing. THis is done via checking the mcu_testing flag.

### 4.1.4 Class method: `start_streaming()`
This method will be executed in the while true loop and it will continuelly loop through all generated vehicles from spawn_vehicle. It will get the needed data and then build the json for the Kafka. It will produce message to the producer. After the for loop is done it will clear the producer buffer and with this push the data to the kafka and restart the for loop.

### 4.1.5 Class method: `on_delivery()`
This is a Kafka callback. It is triggered when Kafka delivers the message. We are just using it to print the error message if it is there.

### 4.1.6 Class method: `_on_obstacle_detected()`
This is the callback that will give the distance till the obstacle. In our case we are checking only upfront. It is using the self.distance dictionary, and based in vehicle.id getting the value out.

### 4.1.7 Class method: `get_lane_data()`
This method is used to get the offset from lanes. This is important part of our dataset, since it is gathering the important data for neural network especially for steering the vehicle.

### 4.1.8 Class method: `get_vehicle_data()`
Method important for testing our model in close loop. It is getting all the data needed for the NN inputs and it return it.

### 4.1.9 Class method: `set_carla_data()`
Method used in close loop testing. It is setting the outputs from NN to it's predictions and end it to car inside Carla environment.

### 4.1.10 Class method: `cleanup()`
Kind of a destructor in our code. It is used for clearning the buffers and closing the connections.

### 4.2 Class: TelemetryConsumer
This class is located in telemetry_consumer.py.
> ### TODO(v0.2.0): flash_buffer need to delete existing csv if there
### 4.2.0 Class Methods: 
* [`__init__()`](#421-constructor-__init__)
* [`kafka_connection()`](#422-class-method-kafka_connection)
* [`kafka_start_listening()`](#423-class-method-kafka_start_listening)
* [`flush_buffer()`](#424-class-method-flush_buffer)
* [`stop_listening()`](#425-class-method-stop_listening)

### 4.2.1 Constructor: `__init__()` 
Constructor used to initialize the variables needed for code execution.

### 4.2.2 Class method: `kafka_connection()` 
Method creates a Kafka consumer and subscribe the conumer to the topic selected in the config.py.

### 4.2.3 Class method: `kafka_start_listening()`
This method is executed in while loop. It is polling for messages. In case of timeout or error it is printing this to the terminal but continue with execution. -> One failure it is not critical. It is saving all messages to buffer and when buffer is equal or bigger than size setted in the config.py it will flush the buffer with help of next method.

### 4.2.4 Class method: `flush_buffer()`
It is getting the buffer and then output the buffer to csv file. It is important to delete csv (if exits) everytime before we start with new data capturing.

### 4.2.5 Class method: `stop_listening()`
It is used to safely close the consumer and flush all buffers.

## 5.0 TinyML models generation
In below section we are explaining from dev point of view methods, but in this case a bit more into details.
At some moment of time during the imoplementation of this code stack we were facing problems that close loop testig was not working as expected. The things we tried are listed at the end of the document with the explanaiton why we did it.

We tried to do this thing close from one point of view so that developer just getting this code can run one class and it is doing everything for him, but on the other hand we wanted to make it as generic as possible so that user can change things. COnfiguration file is there exactly for that reason.

The csv file need to be already generated in this part of the complete environemnt.

Flow is the following:
Our entry point is class named NAS. This one lieves in the nas.py. 
In the main file we need to do the following:

Define the problem
``` python
problem = NAS(config=GLOBAL_CONFIG)
```
NSGA2 is the genetic algorithm from pymoo
``` python
algorithm = NSGA2(pop_size=pop_size)
```
Then we call the minimize method that will start executing the search of best neural network.
``` python
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', n_gen),
    seed=42,
    verbose=True
)
```
So as we see from our code perspective netry point is NAS class, so with this one we will also start here.

### 5.1 Class: NAS
As already stated this is our tnery point to the build of TinyML models.
>### TODO(v0.2.0): Replace simulated CPU latency benchmarking with actual NXP MCXN947 MCU profiling values
>###               or a lookup table based on hardware-in-the-loop (HIL) cycle measurements.
>### TODO(v0.2.0): Add activation functions into the config file and start using count from there.
>### TODO(v0.2.0): Latency weight is potentional rist. We need to use it smartly

### 5.1.0 Class methods
* [`__init__()`](#511-constructor-__init__)
* [`get_model_size_bytes()`](#512-class-method-get_model_size_bytes)
* [`get_activation_function()`](#513-class-method-get_activation_function)
* [`_evaluate()`](#514-class-method-_evaluate)

### 5.1.1 Constructor: `__init__()`
It initialize all needed variables in this class. 
This method also calls DataPreprocesing class to change the dataset data to usefull formats and separation the dataset to train, validation and test dataset.
Here we alsop build kind of dataset. We give min and max number of neurons in layer and also min and max for relu and min and max for layer norm. NAS or pymoo will then take one of those based on the reuslts.
Give special attention to xl and xu. Those two variables and important for NN build.
For each layer we are giving lower and upper boarder. Let' give for example activation function, we know that there are only 3 possibilities and count starts with 0, so that means that we have 3 possibilities 0-2. That is how we set the lower and upper limit.

> Update on normalization: lower and upper limits are both set to 0. We want to avoid normalization at the time being.

### 5.1.2 Class method: `get_model_size_bytes()`
This is a simple method that calculate the size of the model.

### 5.1.3 Class method: `get_activation_function`
Based on the number 0-2 it will return string for the activation function.

### 5.1.4 Class method: `_evaluate()`
Since our NAS class inherits from ElementwiseProblem this method is a must to be overwritten!
By far most important method in this class. It will check the proposed NN and check the if the targets like size, accuaricy etc. are aligned with out proposed limits.
But let's check it into details:
We will get the proposed architecture as an np array. This will contain the num of hidden layers basically we will get to know the num of hidden layers with the proposed number of neurons in the specific layer. Then next argument will be the activation function and last one if layer normalization is included or not.

After that the first check if performed:

``` python
structure_check = [l for l in layer_structure if l > 0]
g1 = -1.0 if len(structure_check) > 0 else 1.0
```
This check will look into layer structure and if all of the hidden layers have the value equal to 0 we will give g1 penalty to 1.0 if all of them are zero (negative value is positive scenario in this case). With this we will prevent that pymoo will give proposals for the NN that is not having and hidden layer

If the structure is valid, we will continue. Next step is to get the size of the NN.
We will build the model with Autonomous driving class.

When we get the size, we will set the g2 error like that:
``` python
g2 = model_size_kb - self.config.ai.max_model_size_kb
```
g1 and g2 are inequality constrains. If g <= 0 success if g > 0 failure.

Then we get the dummy input and switch to torch.no_grad() and we try 30 times to predict out of the model. With that we will get the model latency. We measure the time and then transform to ms.
Then we can calculate the tiny_ml_cost. This cost is vombination of model size and latency. We are using the latency_weight from the configuration to get a bit better results because we are combining kb and ms.
What is more important size of latency?
In case of any error we are setting high penalties.
We store g1 and g2 into:
``` python
out["G"] = [g1, g2]
```

Now we need to move to second penalty that in our case is validation.
We train the model and get the validation loss. We shore everything to:
``` python
out["F"] = [validation_loss, tiny_ml_cost]
```
F penalties are objective functions so basically our main goals what we want to optimize.
In case of any failure, we set high penalties.

### 5.2 Class: DataPreprocessing
This class is located in preprocessing.py
>### TODO(v0.2.0): Refactor smart downsampling logic into a standalone dataset cleaning script 
>###               or dedicated module to separate raw I/O transformation from PyTorch DataLoader assembly.
>### TODO(v0.2.0): Move exploratory method data_preprocessing_view() to a dedicated file.
>### TODO(v0.2.0): Loading of the downsample dataset is at the moment in one big method. Move this code to separate method and put configs into config file.
>### TODO(v0.2.0): Move data_preprocessing_view into helper function since it is not needed anymore.

UPDATE: This doc need to be updated since now we are siplitting datasset based on the vehicle id and timestamp and then separate everything by town.
At the moment Town 1 and Town 2 are used for training, Town 3 for validating and Town 4 for testing.
### 5.2.0 Class methods
* [`__init__()`](#521-constructor-__init__)
* [`read_and_preprocess_dataset()`](#522-class-method-read_and_preprocess_dataset)
* [`save_test_data()`](#523-class-method-save_test_data)
* [`data_preprocessing_view()`](#524-class-method-data_preprocessing_view)

### 5.2.1 Constructor: `__init__()`
Constructor will just initialize the needed variables.

### 5.2.2 Class method: `read_and_preprocess_dataset()`
This method will get the csv file, it will load it and separate it to the train, validation and test dataset. It will also store test dataset for later (we don't want that during the training the model see the test dataset!) and scaler file will also be saved, so we can scale the inputs later with same scaler. This is important otherwise, we will get wrong results.

Let's break this method into more details:

1. We will load the csv file

2. the sort will be performed. We added the sort due to a problem that we are training on x amouth of vehicles. We are storing data to Kafka and later to the csv in following format:
    vehicle 1 data x data x data x time
    vehicle 2 data x data x data x time
    vehicle x ...  ... ... time
    vehicle 1 ... ... ... time
So this results to a problem that we don't have sequnetal data and NN is not aware of a context of driving. This can be a problem since NN will learn based on the data points and will not be aware of a context.
> Another way is also to give as an input 3 data points later -> need to be tested in the code
So comming back, we are then sorting by vehicle_id and timestamp and with that we should get nice dataset that is sequential
>>> IMPORTANT: This step we are avoiding now since we downsample the dataset and nn is not anymore aware of a context!

3. In case of missing values we drop them if they are in the dataset.
4. We check all inputs inside dataset and basically throw out the noise from a sensor etc.
``` python
bounds = getattr(self.config.pipeline, "feature_bounds", {})
    for col, (min_val, max_val) in bounds.items():
        if col in dataset_clean.columns:
            if min_val is not None:
                dataset_clean = dataset_clean[dataset_clean[col] >= min_val]
            if max_val is not None:
                dataset_clean = dataset_clean[dataset_clean[col] <= max_val]
```
``` pyhton
feature_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
    default_factory=lambda: {
        'speed': (0.0, None),
        'acceleration': (0.0, None),
        'distance': (0.0, 50.0),
        'throttle': (0.0, 1.0),
        'brake': (0.0, 1.0),
        'steer': (-1.0, 1.0),
    }
)
```
> Print once again here to see what we get:
5. We then separate the dataset into input adn target_output features and store them to x and y
``` python
X = dataset_clean[self.input_features]
y = dataset_clean[self.target_outputs]
```
6. The next part of the code can cause out problem with steering

We split the X and y values into train and temp values with the followig code:
```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=self.config.pipeline.test_size, 
    random_state=self.config.pipeline.random_state,
    shuffle=True
)
```
In this case test_size is set to 0.2 which means that 80% of data will be stored into X_train and y_train and 20% into temp variables
Then we do
```python
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=self.config.pipeline.validation_size,
    random_state=self.config.pipeline.random_state,
    shuffle=True
)
```
Here the validation size is set to 0.15 which store 17% into X and y validation and rest of 3% into test.
> Shuffle is set to True since we downsample the dataset.

Then we check if scaler already exists or if not we create one.
If scaler exists, we use it and just scale the data, if not then we need to create one, we do this with help of fit_transform. Fit_transform is done only on train data, the rest we just transform with min and max values from the train data.
> We are using RobustScaler. Out dataset is not better since it is having the approx. same amount of data for strainght driving and also for a curves.

7. We build the dataset with PyTorchDataset which is a class in the preprocessing.py. It have only 3 methods:
* `__init__()`: it change from np array to torch.tensor array
* `__getitem__()`: return a single tuple that is equal to specific index
* `__len__()`: Return the total number of samples

8. We save the scaled X_test data to a file. We use this later in the testing.
9. For all three train, validation and test dataset we save them as DataLoader and put the data into batches. Here we can use shuffle true since we want a bit to mix data in batches to prevent overfitting.
10. We return dataloaders.

### 5.2.3 Class method: `save_test_data()`
This method just get the scaled X_test and y values and store it to the file for testing.

### 5.2.4 Class method: `data_preprocessing_view()`
Thie method need to be removed. Check TODO!

### 5.3 Class: AutonomousDriving
This class is located in models.py

>### TODO(v0.2.0): Rename class from AutonomousDriving to DynamicMLP or GenericMLP 
>###               since the dynamic architecture engine is fully dataset-agnostic.
>### TODO(v0.2.0): Refactor output activation splitting in forward() to be dynamic based on config 
>###               (e.g., config-driven activation maps per output index instead of hardcoded 2:3 and 0:2 slices).

### 5.3.0 Class methods
* [`__init__()`](#531-constructor-__init__)
* [`build_network()`](#532-class-method-build_network)
* [`_get_activation_function()`](#523-class-method-save_test_data)
* [`forward()`](#523-class-method-save_test_data)

### 5.3.1 Constructor: `__init__()`
Constructor will just initialize the needed variables. From the constructor, we automatically call build_network calss method.

UPDATE: 6.9.2026:
- We are using GRU first layer to give the network a context regarding previous driving. This helps out network to perform the predictions. More about this we will write in the v0.2.0.

### 5.3.2 Class method: `build_network()`
We have two methods here.
One internal one _build_network -> this one is used to build everything
Then we have build_network -> This one can be called from outsite to change the structure of NN
In this method we are going through the structure of our neural netwrok proposed by NAS and save the items to module_list.
Saving things like nn.LayerNorm etc. If layer normalization is enabled, then we save also those elements into dedicated list.

### 5.3.3 Class method: `_get_activation_function()`:
Out of string we return actuall nn activation function.

### 5.3.4 Class method: `forward()`:
This method is automatically called because the class inherits from the nn.Module.
We are going through all hidden layers and it is doing the following:
X is out input data. Then we put x into layer and perform the calculation, then this x is input for next layer etc.
At the end we also oput activation function for each output, since if we are not doing this steer will get values between -infinity and +infinity but it should be between -1 and 1. Break and throttle should be between 0 and 1 and this is the reason why we use sigmoid. Due to that we need to get the last layer and separate by steer and pedals to apply the activation function and then combine it again.

Example:
> Tensor([50, 10, 2]), layer structure[4,3], outout(a,b)

> In this case:
>Module_list[0] = nn.Linear(3,4) (3 is input testor (50,10,2) and 4 is comming out of layer structure)
>Module_list[1] = nn.Linear(4,3) (4 us from the layer structure and output of MOdule_list[0])
>Module_list[2] = nn.LInear(3,2) (3 is from layer structure and output of MOdule_list[1], 2 is output)

SO the loop will do following:
It will input 50, 10, 2
x is output and we get e.g. (120, -40, 5, 15)
If needed we do normalization
we get: x = (1.58, -1.08, -0.33, -0.17)
we do activation so put negative values to 0
we get: x = (1.58, 0, 0, 0)
THis is input into iteration 2
And we get out:
x=(-15, 50, 10)
NOrmalization
x=(-0.91, 1.34, -0.43)
activation
x=(0, 1.34, 0)
And then output:
[-0.12, 0,75]

### 5.4 Class: ModelsTraining
This class is located in train.py

>### TODO(v0.2.0): Remove or fully deprecate _compute_weighted_loss if Smart Downsampling 
>###               completely resolves the steering class imbalance.


### 5.4.0 Class methods
* [`__init__()`](#541-constructor-__init__)
* [`_compute_weighted_loss()`](#542-class-method-_compute_weighted_loss)
* [`run_training()`](#543-class-method-run_training)

### 5.4.1 Constructor: `__init__()`
Preparing all the values needed and calling AutonomousDriving class to prepare the network.

### 5.4.2 Class method: `_compute_weighted_loss()`
This method was used to calculate the loss in combination with the steer value. We tried to give higher penalty if the road had a curve and with this force the model to learn the steerig better in the curves.
> IMPORTANT NOTE: DEPRECATED

### 5.4.3 Class method: `run_training()`:
In this method we actually do a training. 
We reset the accumulated gradients from previous iteration.
We execute:
```python
outputs = self.model(X_batch)
```
This function actually execute forward function.
After that we calculate loss and perform backpropagation -> compute loss gradients with respect to model parameters.
Then we optimize the wights.

Then we also test on the validation data and output all losses.

Detailed explanation:
We are going through all epochs
We put the model into train mode:
```python
self.model.train()
```
We dont want to accumulate gradients
```python
self.optimizer.zero_grad()
```
This calls forward method with the first batch of data
```python
outputs = self.model(X_batch)
```

Compute the loss
```python
loss_value = self.loss(outputs, y_batch)
```

For each weight we calculate how much it added to error (gradient)
```python
loss_value.backward()
```

It checks the gradients and change the weights
```python
self.optimizer.step()
```
learning_rate = how big the step is that Adam optimizar can do when updating the weights.
Later we are switchig the model to validation to see how it behaves.

### 5.5 Class: ExportEngine
This class is located in export_engine.py
Since we have a problem with 
>### TODO(v0.2.0): act_map = {0: "relu", 1: "tanh", 2: "sigmoid"} -> This need to be updated or taken from the conf file
>### TODO(v0.2.0): Isolate Keras/TensorFlow model conversion into a separate sub-process 
>###               (e.g., via Python multiprocessing) to resolve PyTorch and TensorFlow 
>###               C++ runtime (libgomp/OpenMP) import conflicts cleanly.

### 5.5.0 Class methods
* [`__init__()`](#551-constructor-__init__)
* [`verify_models()`](#552-class-method-verify_models)
* [`representative_data_gen()`](#553-class-method-representative_data_gen)
* [`copy_weights()`](#554-class-method-copy_weights)
* [`export_tflite()`](#555-class-method-export_tflite)
* [`export_c_header()`](#556-class-method-export_c_header)
* [`export_top_models()`](#557-class-method-export_top_models)

### 5.5.1 Constructor: `__init__()`
Inhere we prepare all variables needed.

### 5.5.2 Class method: `verify_models()`
This method perform clross-framework verification. Ww test pytorch, tensorflow and tflite models.
TEsting is done on one batch only, since the main reason for having this method is to check that the weights were copied correctly.

### 5.5.3 Class method: `representative_data_gen()`:
TFLite need to get a bit of context to convert from float to int numbers -128 do 127. WIth this method we give him tge values one by one.
100 values is enought.

### 5.5.4 Class method: `copy_weights()`:
With this method we are basically just coping the weights  to tf model.
Very important in this code is that pytourch store the wieghts in the flow [out_features, in_features], but keras in [in_features, out_features].
That is why we are using .T.
We also need to set the output layer correctly, that is why we are correctly setting the values.
### 5.5.5 Class method: `export_tflite()`:
Method exports the model to tflite format. Here we need representative_data_gen method. All hidden layers are set as int8 so we are using quantization, but input and output layers stays at the float values.

### 5.5.6 Class method: `export_c_header()`:
We take the tflite and convert the model to .h file. This one is then used in the MCU. we more or less open the tflite file and take a bytes out and print it into the .h file.
### 5.5.7 Class method: `export_top_models()`:
We are going through all the models that are accepted by pymoo. We train the model since the pymoo is storing only how the model is build but not actual weights etc. THen we export it to the needed files.

## 6.0 Testing
We have possibility to test close loop with carla and also prediction by MCU and then check if everyhting is okay.
Test class can be found in tinyML_CPU_execution. There is a file execute_on_mcu.py. This file containst the ExecuteOnMCU class.

In both testing methods the test is performed with NXP MCU.


>### TODO(v0.2.0): When sending to the MCU we are using fixed values also checking in receiving we hardcoded 12, this need to be updated.

### 6.1 Class methods
* [`__init__()`](#62-constructor-__init__)
* [`close_loop_testing_CARLA()`](#63-constructor-close_loop_testing_carla)
* [`_send_to_mcu()`](#64-constructor-_send_to_mcu)
* [`test_on_dataset()`](#65-constructor-test_on_dataset)
* [`load_dataset()`](#66-constructor-load_dataset)

### 6.2 Constructor: `__init__()`
We are just preparing the environment for testing.
### 6.3 Constructor: `close_loop_testing_CARLA()`
Here we are testing the model on Carla directly. We get the data from the get_vehicle_data and scale it. Then we send it to the MCU and received data we send to the carla and steer the car with it.
### 6.4 Constructor: `_send_to_mcu()`
FUnction used to send data via Serial to the MCU. We add the header to the payload. We return the output from MCU.
### 6.5 Constructor: `test_on_dataset()`
More or less same story as close loop testing. Only difference is that here we are not controlling the car is Carla, but we are only comparing the outputs with test data.
### 6.6 Constructor: `load_dataset()`
We are loading the scaler and the testdata.npz.

## 7.0 Code on the MCU
TODO:

## 8.0 Helper functions
We add folder named debugging script. Inside this one we added the things that helped us especially with preprocessing of the data.
In those files user can also find the examples of the outputs and then actions taken.

## 9.0 Problems and solutions
> Problem description: steering need to be between -1.0 and 1.0. Break and velocity need to be between 0 and 1.
> Solution: We put to the output layer to different activation functions

> Problem description: Since we are getting the data with the autopilot, we have a problems that cars are not startig to drive
> Solution: Not there yet


* We remove the friction from inputs since it was set to 0.1
* Check the todo in some files to see what we did

* MinMaxScaler replaced by StandardScaler -> reason is that for lane parts we got extremly low values and minmaxscaler together with putting the inputs to int8 completly kill this data. We see positive results with those changes, car is driving nicelly in the straingt part. We also add a bit of the code that id predicted value is smaller than 0.08 we put the steering to 0.0
* Now we increese also weight for the part where turn is due to big amount of data where car is driving in the straight way. It is not workign and we also remove this since we downsample the dataset.

> Problem: We had a tight coupling between classes especially in ml_pipeline
We solve this problem with factory functions instead of directly calling or creating an instance of a class. This lead to cleaner code etc.
In NAS you will see that we are calling both model and train factory in a loop and there we are loading model to the train directly.
Not all data is mandatory to be added, especially train, validation and test dataset are not mandatory.
Also on carla streaming in execute on mcu, there the carla is still optional, since if it is not added it will simply create a new instance


## 10.0 Tests

We are using pytest to execute tests.

test_global_config_structure -> Tests that specific values are having valid entries : This test is a test test :D so that we see the test env is working
test_autonomous_driving_model_shape -> We are checking that the output from the model is correct
test_model_factory_pattern -> We are checking that model factory correctyl create a AutonomousDriving instance

## 11.0 GUI
It is developed in NiceGUI with major help of the Google Gemini!

For GUI we add the class into nas.py:
NASHistoryCallback -> it is used as a callback and responsible for saving the data for each generation into json file.
This is on our site strictly used for the graphs outputs.

notify -> this method is overwritten from Callback class.
We get the needed data and then output the graphs from json file.

GUI is used to nicely show or give the user possibility to run the scripts from the GUI and at the end of a training also get some graphs to see what is going on.
To get the GUI up, execute: python dashboard.py

Current diagrams:
Pareto Trade-off Chart: Shows pareto front models in comparison between validation loss and size in kB (size of TFLite model)
Model Footprint COmparison: Shows the difference between the size of the PYtorch model and TFLite model
Quantization error: Shows the difference in Mean ABsolute Error between Pytorch model and TFLite model.
    We are using 1 batch of data to see if the model after transformation dont increese the error

Inside NAS convergence history we are showing the graph of all pareto front networks and where they live in comparison between loss and cost.
And on the second grapg got each generation we show the average loss and best loss.

## 12.0 Debugging scripts
At the moment we are releasing also debugging scripts. Here we are performing the tests on the dataset and building basic networks for testing. We will make this generic as possible and at the end plan is to have dedicated module for dataset review.