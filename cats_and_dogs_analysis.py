import os
import hpo

bayes_results = hpo.Results.load(os.path.join(os.getcwd(), "cats_and_dogs_bayesian_random_forest.results"))
ga_results = hpo.Results.load(os.path.join(os.getcwd(), "cats_and_dogs_ga_roulette.results"))
rs_results = hpo.Results.load(os.path.join(os.getcwd(), "cats_and_dogs_hpo_random_search.results"))
#bayes_results.plot_average_score_over_optimisation_period()

print(bayes_results.best_result().model_configuration().hyperparameter_values())
print(ga_results.best_result().model_configuration().hyperparameter_values())
print(rs_results.best_result().model_configuration().hyperparameter_values())