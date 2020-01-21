import hpo
import hpo_genetic_algorithm
import ray
import os
import cats_and_dogs_data as data

class CatsAndDogsChromosome(hpo_genetic_algorithm.Chromosome):
    def __init__(self, model):
        super().__init__()

        self._model = model
        self._genes = list()
        for layer in ray.get(self._model.layers.remote()):
            for hyperparamater in layer.hyperparameters():
                self._genes.append(hpo_genetic_algorithm.Gene(layer.layer_name() + "_" + hyperparamater.name(), hyperparamater.value_range(), hyperparamater.value(), hyperparamater.constraints()))

    def check_gene_constraints(self, gene):
        return True

    #train(self, training_dataset, training_steps, validation_dataset=None, validation_steps=0, number_of_epochs=10, cached_model_path=None, history_path=None) -> dict:
    def execute(self):
        layers_result_id = self._model.layers.remote()
        layers = ray.get(layers_result_id)
        for layer in layers:
            for hyperparameter in layer.hyperparameters():
                gene_name = layer.layer_name() + "_" + hyperparameter.name()
                gene = next(x for x in self._genes if x.name() == gene_name)
                #will throw if no gene is found with that name
                hyperparameter.value(gene.value())

        self._model.layers.remote(layers)

        history_result_id = self._model.train.remote()
        history = ray.get(history_result_id)
        validation_accuracy = history["val_accuracy"][-1]
        self._fitness = int(validation_accuracy * 1000)

    def decode(self):
        genes = list()
        layers_result_id = self._model.layers.remote()
        layers = ray.get(layers_result_id)
        for layer in layers:
            for hyperparameter in layer.hyperparameters():
                gene_name = layer.layer_name() + "_" + hyperparameter.name()
                genes.append((gene_name, hyperparameter.value()))

        return genes