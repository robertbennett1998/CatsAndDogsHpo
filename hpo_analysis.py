import json
import os
import matplotlib.pyplot as plt

filepaths = [os.path.join(os.getcwd(), "generation_history_08_Mar_2020__11_02_36.json")]

legend = list()
i = 0

for filepath in filepaths:
    avg_fitnesses = list()
    best_fitnesses = list()

    legend.append("Average Fitness " + str(i))
    legend.append("Best Fitness " + str(i))

    gen_history = json.load(open(filepath))
    for population in gen_history:
        avg_fitnesses.append(population["avgerage_population_fitness"])
        best_fitnesses.append(population["best_fitness"])

    plt.plot(avg_fitnesses)
    plt.plot(best_fitnesses)
    i += 1

print(best_fitnesses)

plt.legend(legend)
plt.xlabel("Generation")
plt.ylabel("Fitness (accuracy)")
plt.title("Average fitness and best fitness over generations")
plt.show()
