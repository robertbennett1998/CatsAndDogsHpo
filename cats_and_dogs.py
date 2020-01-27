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

ray.init()

model_layers = [
    #0 - Conv 2D - Optimise
    hpo.Layer(layer_name="input_layer", layer_type=tf.keras.layers.Conv2D, 
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=16, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
        hpo.Parameter(parameter_name="input_shape", parameter_value=(200, 200, 3))
    ]),

    #1 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_1", layer_type=tf.keras.layers.MaxPooling2D, 
    hyperparameters=[], 
    parameters=[]),

    #2 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_2", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=32, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #3 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_3", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[], 
    hyperparameters=[]),

    #4 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_4", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=64, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #5 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_5", layer_type=tf.keras.layers.MaxPooling2D, 
    hyperparameters=[], 
    parameters=[]),

    #6 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_6", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(parameter_name="filters", parameter_value=128, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(parameter_name="kernel_size", parameter_value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(parameter_name="padding", parameter_value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #7 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_7", layer_type=tf.keras.layers.MaxPooling2D, 
    hyperparameters=[], 
    parameters=[]),

    #8 - Flatten - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_8", layer_type=tf.keras.layers.Flatten, 
    hyperparameters=[], 
    parameters=[]),

    #9 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_9", layer_type=tf.keras.layers.Dropout, 
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ]),

    #10 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_10", layer_type=tf.keras.layers.Dense, 
    hyperparameters=[
        hpo.Parameter(parameter_name="units", parameter_value=512, value_range=[2**x for x in range(2, 10)], constraints=None),#range between 4 and 512
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
    ]),

    #11 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_11", layer_type=tf.keras.layers.Dropout, 
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2, value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ]),

    #12 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_12", layer_type=tf.keras.layers.Dense, 
    parameters=[], 
    hyperparameters=[
        hpo.Parameter(parameter_name="units", parameter_value=256, value_range=[2**x for x in range(2, 10)], constraints=None),#range between 4 and 512
        hpo.Parameter(parameter_name="activation", parameter_value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #13 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_12", layer_type=tf.keras.layers.Dropout, 
    hyperparameters=[
        hpo.Parameter(parameter_name="rate", parameter_value=0.2 , value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(parameter_name="seed", parameter_value=42)
    ]),

    #14 - Dense - Optimise
    hpo.Layer(layer_name="output_layer", layer_type=tf.keras.layers.Dense, 
    hyperparameters=[
        hpo.Parameter(parameter_name="activation", parameter_value="sigmoid", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
        hpo.Parameter(parameter_name="units", parameter_value=2)
    ])]
model_configuration = hpo.ModelConfiguration(layers=model_layers)

def construct_cats_and_dogs_data():
    return cats_and_dogs_data.CatsAndDogsData(os.path.join(os.getcwd(), ".cache"), True, True, True, 100, 100, 100)

def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)

strategy = hpo_strategy_ga.GeneticAlgorithm(population_size=20, max_iterations=10, chromosome_type=construct_chromosome)
strategy.mutation_strategy().mutation_probability(0.6)
strategy.survivour_selection_strategy().threshold(0.7)

hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy)

hpo_instance.execute()