import cats_and_dogs_data
import os
import hpo
import hpo.strategies.genetic_algorithm
import hpo.strategies.bayesian_method
import hpo.strategies.random_search
import hpo_experiment_runner
import cats_and_dogs_models

data_dir = "/home/rob/dissertation/data/dogs-vs-cats/train/"

def construct_cats_and_dogs_data():
    return cats_and_dogs_data.CatsAndDogsData(data_dir, os.path.join(os.getcwd(), ".cache"), True, True, True, 50, 50, 50)


def construct_chromosome():
    return hpo.strategies.genetic_algorithm.DefaultChromosome(model_configuration)


def model_exception_handler(e):
    print("Exception occured while training the model.", e)

model_configuration = hpo.DefaulDLModelConfiguration(optimiser=cats_and_dogs_models.optimiser, layers=cats_and_dogs_models.cnn, loss_function="categorical_crossentropy", number_of_epochs=10)

####################################
# Random Search
####################################
strategy = hpo.strategies.random_search.RandomSearch(model_configuration, 100)
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_random_search.results"))

#####################################
# Bayesian Selection - Gaussian Process
#####################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 100, hpo.strategies.bayesian_method.GaussianProcessSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_bayesian_gaussian_process.results"))

#####################################
# Bayesian Selection - Random Forest
#####################################
strategy = hpo.strategies.bayesian_method.BayesianMethod(model_configuration, 100, hpo.strategies.bayesian_method.RandomForestSurrogate())
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_bayesian_random_forest.results"))

########################################
# Genetic Algorithm - Threshold Selection
#########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=10, max_iterations=10, chromosome_type=construct_chromosome, survivour_selection_stratergy="roulette")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().survivour_percentage(0.7)
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_genetic_algorithm_roulette.results"))

########################################
# Genetic Algorithm - Roulette Selection
#########################################
strategy = hpo.strategies.genetic_algorithm.GeneticAlgorithm(population_size=10, max_iterations=10, chromosome_type=construct_chromosome, survivour_selection_stratergy="threshold")
strategy.mutation_strategy().mutation_probability(0.05)
strategy.survivour_selection_strategy().threshold(0.9)
hpo_instance = hpo.Hpo(model_configuration, construct_cats_and_dogs_data, strategy, model_exception_handler=model_exception_handler)

hpo_experiment_runner.run(hpo_instance, os.path.join(os.getcwd(), "cats_and_dogs_hpo_genetic_algorithm_roulette.results"))
