import hpo
import os

#results = hpo.Results.load(os.path.join(os.getcwd(), "cats_and_dogs_bayesian_random_forest.results")) 
results = hpo.Results.load(os.path.join(os.getcwd(), ".tmp/hpo.results.tmp"))
print(results.history()[-1].meta_data())
print(results.best_result().score())
#results.plot_average_score_over_optimisation_period()
