# Intro
We decide that from the complete pipeline we will start working on the making a TinyML framework our of the project.
Idea is that after one year we can came back and use different dataset and just rewrite needed methods and the proces is done.
In this document we will list down all changes and why are needed and at the end also explain how to use the first version of framework.

# Structure changes

We rename ml_pipeline folder into core.
Inside core we create following folders:
* configuration
* data_preprocessing
* export
* model
* nas
* train

Inside them we added dedicated file that is at the moment matching the previous implementation, but they will be rewritten and documented here.

test folder also changed. We added into test folder following structure:
* tests
  * core
    * configuration
    * export
    * train
    * etc.

So we are basically mapping the core format.

Decision was taken that we will do following:
Inside core/configuration we added 'base_configuration.py'. Inside this file we have default settings that are same always also if customer is comming with different dataset.
Those settings are not allowed to be rewritten in this file.
Each user can do:
Add dedicated use-case into use_cases folder in root (carla_driving is example).
Inside this folder then user can add .py file and inside they can add dedicated settings and also change the default settings. (check carla_settings.py) for example.
This .py file with final class is inherit from our setting and this is a must thing to do.


The tests for those changes are added.

## Data preparation changes
At the moment we decide that the first step and release for 0.2.0 will be only separation between core and use_case code. So at the moment we are not introducing multiple interfaces, but just one method that then need to be rewriten in the use_case.
In core/data_processing we added two classes:
* base_dataset.py : BaseDataset -> this one is just for creading pytorch tensor and it is provided at bare minumum at the moment -> TODO: Need to be tested for other datasets
* base_provider.py: BaseDataProvider -> Introduces at the moment prepare_dataset method that need to be rewritten for each usecase.

User can then add into use_cases/their_usecase/ a file for example: carla_provider.py that implements prepare_dataset and return the 3 objects (train, validation and test) datasets. In future we will break out method to multiple methods to have nicer coding standard.

## Model change in core:
We basically just make two interfaces _build_network and forward. Those methods will be moved to use_cases/carla_driving/carla_models.py.
There inside we actually move both methods. As said already few times now, at the moment we are just separating the interfaces and that is it.
Some basic tests are also added.

## Data and exported models:
Data folder need to be part of specific usecase, that is also a reason that we move it there now. Also exported models need to be now in the specific use_case folder.

## Train update:
We added a test and comment a bit better the train. But run_training is a common method that should be used across multiple usecases that is why we didn't rewrite it and we didnt add any interface inside.

## NAS
We switch also here to a bit different thing. User need to give xl and xu into the basenas, with this we gain that there is unlimited amount of boundaries inside. User also need to rewrite the _evaluate method, since evaluation is also a different between usecases. _decode method will build a nice view of the search_space.
In this class we always need to add model and training, since the NAS in dynamically build the model and test it.