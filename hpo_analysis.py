import json
import os
import matplotlib.pyplot as plt

gen_history_path = os.path.join(os.getcwd(), "generation_history_28_Jan_2020__18_04_59.json")

gen_history = json.load(open(gen_history_path))

avg_fitnesses = list()
best_fitnesses = list()

for population in gen_history:
    avg_fitnesses.append(population["avgerage_population_fitness"])
    best_fitnesses.append(population["best_fitness"])


plt.plot(avg_fitnesses)
plt.plot(best_fitnesses)
plt.legend(["Average fitness", "Best fitness"])
plt.xlabel("Generation")
plt.ylabel("Fitness (accuracy)")
plt.title("Average fitness and best fitness over generations")
plt.show()