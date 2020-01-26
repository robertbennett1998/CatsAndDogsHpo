import tensorflow as tf
import os
from pathlib import Path
import data_preprocessing

def get_datasets(img_width, img_height, training_batch_size, validation_batch_size, test_batch_size, cache_path, augment_training_data, augment_validation_data, augment_test_data, class_labels):
    def get_jpeg_from_filepath(path):
        def get_class_label_from_filepath(path):
            filename = tf.strings.split(path, os.path.sep)
            return tf.strings.split(filename[-1], '.')[0]

        class_label = get_class_label_from_filepath(path)
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_width, img_height])
        return img, class_label == class_labels

    def flip_image_y(img, label):
        return tf.image.flip_up_down(img), label

    def flip_image_x(img, label):
        return tf.image.flip_left_right(img), label

    def flip_image_xy(img, label):
        return tf.image.flip_left_right(tf.image.flip_up_down(img)), label

    training_files, validation_files, test_files, training_image_count, validation_image_count, test_image_count = data_preprocessing.create_filepath_datasets_from_directory(os.path.join(os.getcwd(), "../../data/dogs-vs-cats/train/"), "*.jpg")

    training_images = data_preprocessing.transform_dataset(training_files, get_jpeg_from_filepath)
    validation_images = data_preprocessing.transform_dataset(validation_files, get_jpeg_from_filepath)
    test_images = data_preprocessing.transform_dataset(test_files, get_jpeg_from_filepath)

    if augment_training_data:
        augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_x)
        augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_y, augmented_training_images)
        augmented_training_images = data_preprocessing.augment_dataset(training_images, flip_image_xy, augmented_training_images)
        training_images = augmented_training_images
        training_image_count *= 4

    training_images = data_preprocessing.prepare_dataset(training_images, training_batch_size, cache=os.path.join(cache_path, "training_images.tfcache"))

    if augment_validation_data:
        augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_x)
        augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_y, augmented_validation_images)
        augmented_validation_images = data_preprocessing.augment_dataset(validation_images, flip_image_xy, augmented_validation_images)
        validation_images = augmented_validation_images
        validation_image_count *= 4

    validation_images = data_preprocessing.prepare_dataset(validation_images, validation_batch_size, cache=os.path.join(cache_path, "validation_images.tfcache"))

    if augment_test_data:
        augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_x)
        augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_y, augmented_test_images)
        augmented_test_images = data_preprocessing.augment_dataset(test_images, flip_image_xy, augmented_test_images)
        test_images = augmented_test_images
        test_image_count *= 4
        
    test_images = data_preprocessing.prepare_dataset(test_images, test_batch_size, cache=os.path.join(cache_path, "test_images.tfcache"))

    return training_images, validation_images, test_images, training_image_count, validation_image_count, test_image_count