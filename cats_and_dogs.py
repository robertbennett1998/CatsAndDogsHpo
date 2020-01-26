import cats_and_dogs_model
import cats_and_dogs_chromosome
import hpo
import hpo_genetic_algorithm
import tensorflow as tf
import numpy as np
import ray
import json
import os
from datetime import datetime

ray.init()

model_layers = [
    #0 - Conv 2D - Optimise
    hpo.Layer(layer_name="input_layer", layer_type=tf.keras.layers.Conv2D, 
    hyperparameters=[
        hpo.Parameter(name="filters", value=16, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(name="kernel_size", value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(name="padding", value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
        hpo.Parameter(name="input_shape", value=(200, 200, 3))
    ]),

    #1 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_1", layer_type=tf.keras.layers.MaxPooling2D, 
    hyperparameters=[], 
    parameters=[]),

    #2 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_2", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(name="filters", value=32, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(name="kernel_size", value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(name="padding", value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #3 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_3", layer_type=tf.keras.layers.MaxPooling2D, 
    parameters=[], 
    hyperparameters=[]),

    #4 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_4", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(name="filters", value=64, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(name="kernel_size", value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(name="padding", value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #5 - Max Pooling 2D - No Optimisation Currently - TODO
    hpo.Layer(layer_name="hidden_layer_5", layer_type=tf.keras.layers.MaxPooling2D, 
    hyperparameters=[], 
    parameters=[]),

    #6 - Conv 2D - Optimise
    hpo.Layer(layer_name="hidden_layer_6", layer_type=tf.keras.layers.Conv2D, 
    parameters=[],
    hyperparameters=[
        hpo.Parameter(name="filters", value=128, value_range=[2**x for x in range(1, 7)], constraints=None),# range from 2-128
        hpo.Parameter(name="kernel_size", value=3, value_range=range(2, 11), constraints=None),#kernal size range from 2 to 10
        hpo.Parameter(name="padding", value="same", value_range=["same"], constraints=None),#need to add more
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
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
        hpo.Parameter(name="rate", value=0.2, value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(name="seed", value=42)
    ]),

    #10 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_10", layer_type=tf.keras.layers.Dense, 
    hyperparameters=[
        hpo.Parameter(name="units", value=512, value_range=[2**x for x in range(2, 10)], constraints=None),#range between 4 and 512
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
    ]),

    #11 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_11", layer_type=tf.keras.layers.Dropout, 
    hyperparameters=[
        hpo.Parameter(name="rate", value=0.2, value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(name="seed", value=42)
    ]),

    #12 - Dense - Optimise
    hpo.Layer(layer_name="hidden_layer_12", layer_type=tf.keras.layers.Dense, 
    parameters=[], 
    hyperparameters=[
        hpo.Parameter(name="units", value=256, value_range=[2**x for x in range(2, 10)], constraints=None),#range between 4 and 512
        hpo.Parameter(name="activation", value="relu", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ]),

    #13 - Dropout - Optimise
    hpo.Layer(layer_name="hidden_layer_12", layer_type=tf.keras.layers.Dropout, 
    hyperparameters=[
        hpo.Parameter(name="rate", value=0.2 , value_range=np.arange(0.0, 0.5, 0.01).tolist(), constraints=None)
    ], 
    parameters=[
        hpo.Parameter(name="seed", value=42)
    ]),

    #14 - Dense - Optimise
    hpo.Layer(layer_name="output_layer", layer_type=tf.keras.layers.Dense, 
    hyperparameters=[
        hpo.Parameter(name="activation", value="sigmoid", value_range=["relu", "tanh", "sigmoid"], constraints=None)#need to add more
    ], 
    parameters=[
        hpo.Parameter(name="units", value=2)
    ])
]

model = cats_and_dogs_model.CatsAndDogsCNN.remote(model_layers)

def create_cats_and_dogs_chromosome():
    return cats_and_dogs_chromosome.CatsAndDogsChromosome(model)

hpo_stratergy = hpo_genetic_algorithm.GeneticAlgorithm(10, 15, create_cats_and_dogs_chromosome)
hpo_instance = hpo.Hpo(hpo_stratergy, model)
hpo_instance.execute()