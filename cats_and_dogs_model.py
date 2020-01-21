import hpo
import ray
import os
import cats_and_dogs_data as data
import matplotlib.pyplot as plt

@ray.remote(num_gpus=1)
class CatsAndDogsCNN(hpo.Model):
    def __init__(self, layers=None):
        super().__init__()

        self._layers = layers
        self._model = None

    def train(self):
        self.build()
        self._model.summary()
        img_width = 200
        img_height = 200
        class_labels = ["cat", "dog"]
        training_batch_size = 250
        validation_batch_size = 250
        test_batch_size = 250
        cache_path = os.path.join(os.getcwd(), ".cache")
        number_of_epochs = 10

        augment_training_data = True
        augment_validation_data = True
        augment_test_data = True
        rebuild_model = True
        model_path = None

        if augment_training_data:
            model_path = os.path.join(cache_path, "augmented_model.tfmodel")
        else:
            model_path = os.path.join(cache_path, "model.tfmodel")

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        training_images, validation_images, test_images, training_image_count, validation_image_count, test_image_count = data.get_datasets(img_width, img_height, training_batch_size, validation_batch_size, test_batch_size, cache_path, augment_training_data, augment_validation_data, augment_test_data, class_labels)

        # training_image_count = 100
        # training_images = training_images.take(training_image_count)

        # validation_image_count = 100
        # validation_images = validation_images.take(validation_image_count)
        
        # for img, label in validation_images.take(1):
        #     for n in range(10):
        #         plt.figure(figsize=(200, 200))
        #         plt.imshow(img[n])
        #         plt.title(label[n].numpy())
        #         plt.show()

        return super()._train(training_images, training_image_count//training_batch_size, validation_images, validation_image_count//validation_batch_size)

    def build(self):
        import tensorflow as tf
        self._model = tf.keras.models.Sequential()
        for layer in self._layers:
            self._model.add(layer.build())
        self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])