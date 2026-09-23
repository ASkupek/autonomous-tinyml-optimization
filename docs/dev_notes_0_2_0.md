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