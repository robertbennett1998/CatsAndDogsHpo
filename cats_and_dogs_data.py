import tensorflow as tf
import os
from pathlib import Path
import data_preprocessing
import hpo

class CatsAndDogsData(hpo.Data):
    def __init__(self, cache_path, augment_training_data, augment_validation_data, augment_test_data, training_batch_size, validation_batch_size, test_batch_size):
        super().__init__()
        self._augment_training_data = augment_training_data
        self._augment_validation_data = augment_validation_data
        self._augment_test_data = augment_test_data
        self._cache_path = cache_path
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
        self._test_batch_size = test_batch_size
        
        self._class_labels = ["cat", "dog"]
        self._img_width = 200
        self._img_height = 200

        self._training_image_count = 0
        self._validation_image_count = 0
        self._test_image_count = 0
        
        self._training_data = None
        self._valdiation_data = None
        self._test_data = None

        if not os.path.exists(self._cache_path):
            os.mkdir(self._cache_path)

    def load(self):
        def get_jpeg_from_filepath(path):
            def get_class_label_from_filepath(path):
                filename = tf.strings.split(path, os.path.sep)
                return tf.strings.split(filename[-1], '.')[0]

            class_label = get_class_label_from_filepath(path)
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self._img_width, self._img_height])
            return img, class_label == self._class_labels

        def flip_image_y(img, label):
            return tf.image.flip_up_down(img), label

        def flip_image_x(img, label):
            return tf.image.flip_left_right(img), label

        def flip_image_xy(img, label):
            return tf.image.flip_left_right(tf.image.flip_up_down(img)), label

        training_files, validation_files, test_files, self._training_image_count, self._validation_image_count, self._test_image_count = data_preprocessing.create_filepath_datasets_from_directory(os.path.join(os.getcwd(), "../../data/dogs-vs-cats/train/"), "*.jpg")

        training_images = data_preprocessing.transform_dataset(training_files, get_jpeg_from_filepath)
        validation_images = data_preprocessing.transform_dataset(validation_files, get_jpeg_from_filepath)
        test_images = data_preprocessing.transform_dataset(test_files, get_jpeg_from_filepath)

        if self._augment_training_data:
            augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_x)
            augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_y, augmented_training_images)
            augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_xy, augmented_training_images)
            training_images = augmented_training_images
            self._training_image_count *= 4

        self._training_data = data_preprocessing.prepare_dataset(training_images, self._training_batch_size, cache=os.path.join(self._cache_path, "training_images.tfcache"))

        if self._augment_validation_data:
            augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_x)
            augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_y, augmented_validation_images)
            augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_xy, augmented_validation_images)
            validation_images = augmented_validation_images
            self._validation_image_count *= 4

        self._valdiation_data = data_preprocessing.prepare_dataset(validation_images, self._validation_batch_size, cache=os.path.join(self._cache_path, "validation_images.tfcache"))

        if self._augment_test_data:
            augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_x)
            augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_y, augmented_test_images)
            augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_xy, augmented_test_images)
            test_images = augmented_test_images
            self._test_image_count *= 4
            
        self._test_data = data_preprocessing.prepare_dataset(test_images, self._test_batch_size, cache=os.path.join(self._cache_path, "test_images.tfcache"))

        # self._training_image_count = 5000
        # self._training_data = self._training_data.take(self._training_image_count)


        # self._validation_image_count = 2000
        # self._valdiation_data = self._valdiation_data.take(self._validation_image_count)

    def training_steps(self):
        return self._training_image_count // self._training_batch_size

    def validation_steps(self):
        return self._validation_image_count // self._validation_batch_size

    def test_steps(self):
        return self._test_image_count // self._test_batch_size

    def training_data(self):
        return self._training_data

    def validation_data(self):
        return self._valdiation_data

    def test_data(self):
        return self._test_data