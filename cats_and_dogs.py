import cats_and_dogs_data
import tensorflow as tf
import numpy as np
import ray
import json
import os
from datetime import datetime
import hpo
import hpo.strategies.genetic_algorithm as hpo_strategy_ga
import ray

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

print(tf.test.is_gpu_available())

ray.init()

cache_path="/home/596616/dissertation/src/CatsAndDogsHpo/.cache"

model_layers = [
    #0 - Conv 2D - Optimise
    hpo.Layer(layer_name="input_layer_conv_2d", layer_type=tf.keras.layers.Conv2D, 
    parameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=16, value_range=[2**x for x in range(1, 9)], constraints=None),# range from 2-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None),
        hpo.Parameter(parameter_name="input_shape", parameter_value=(200, 200, 3))
    ],
    hyperparameters=[
    ]),

    #1 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_1_max_pooling", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[],
    hyperparameters=[]),

    #2 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_2_conv_2d", layer_type=tf.keras.layers.Conv2D, 
    parameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=32, value_range=[2**x for x in range(1, 9)], constraints=None),# range from 2-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)
    ],
    hyperparameters=[
    ]),

    #3 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_3_max_pooling", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[], 
    hyperparameters=[]),

    #4 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_4_conv_2d", layer_type=tf.keras.layers.Conv2D, 
    parameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=64, value_range=[2**x for x in range(1, 9)], constraints=None),# range from 2-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)
    ],
    hyperparameters=[

    ]),

    #5 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_5_max_pooling", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[],
    hyperparameters=[]),

    #6 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_6_conv_2d", layer_type=tf.keras.layers.Conv2D, 
    parameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=128, value_range=[2**x for x in range(1, 9)], constraints=None),# range from 2-512
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 11
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same", "valid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ],
    hyperparameters=[
    ]),

    #7 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_7_max_pooling", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[],
    hyperparameters=[
    ]),

    #8 - Flatten - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_8_flatten", layer_type=tf.keras.layers.Flatten, 
    parameters=[],
    hyperparameters=[]),

    #9 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_9_dropout", layer_type=tf.keras.layers.Dropout, 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ], 
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.4, 0.1).tolist(), constraints=None)
    ]),

    #10 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_10_dense", layer_type=tf.keras.layers.Dense, 
    parameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="units", parameter_value=512, value_range=[2**x for x in range(7, 11)], constraints=None),#range between 4 and 512
    ]),

    #11 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_11_dropout", layer_type=tf.keras.layers.Dropout, 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.4, 0.1).tolist(), constraints=None)
    ]),

    #12 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_12_dense", layer_type=tf.keras.layers.Dense, 
    parameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    hyperparameters=[
        hpo.Parameter(parameter_name="units", parameter_value=256, value_range=[2**x for x in range(6, 10)], constraints=None),#range between 4 and 512
    ]),

    #13 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_12_dropout", layer_type=tf.keras.layers.Dropout, 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ],
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2 , value_range=np.arange(0.0, 0.4, 0.1).tolist(), constraints=None)
    ]),

    #14 - Dense - Optimise
    hpo.Layer(layer_name="output_layer_dense", layer_type=tf.keras.layers.Dense, 
    parameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="sigmoid", value_range=["tanh", "sigmoid"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="units", parameter_value=2)
    ], 
    hyperparameters=[
    ])]

optimiser = hpo.Optimiser(optimiser_name="optimiser_adam", optimiser_type=tf.keras.optimizers.Adam, hyperparameters=[
    hpo.Parameter(parameter_name="learning_rate", parameter_value=0.001, value_range=[1 * (10 ** n) for n in range(0, -7, -1)]) #np.arange(0.0001, 1.0, 0.0005).tolist())
])

model_configuration = hpo.ModelConfiguration(optimiser=optimiser, layers=model_layers, number_of_epochs=10)
print(model_configuration.number_of_hyperparameters())
model_configuration.hyperparameter_summary(True)

def construct_cats_and_dogs_data():
    return cats_and_dogs_data.CatsAndDogsData(cache_path, True, True, True, 100, 100, 100)

def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)

strategy = hpo_strategy_ga.GeneticAlgorithm(population_size=30, max_iterations=8, chromosome_type=construct_chromosome, survivour_selection_stratergy="threshold")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().threshold(0.7)
 
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy)

hpo_instance.execute()
