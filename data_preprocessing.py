import tensorflow as tf
import os
from pathlib import Path

def create_filepath_datasets_from_directory(data_dir, pattern, training_split=0.7, validation_split=0.15, test_split=0.15, randomise_files=True, randomise_files_seed=42):
    if training_split + validation_split + test_split != 1:
        raise Exception("Sum of training, validation and test splits must be one")

    file_count = len(list(Path(data_dir).glob(pattern)))
    training_file_count = int(file_count * training_split)
    validation_file_count = int(file_count * validation_split)
    test_file_count = int(file_count * test_split)
    
    try:
        full_filepaths_ds = tf.data.Dataset.list_files(str(os.path.join(data_dir, pattern)), seed=randomise_files_seed, shuffle=randomise_files)
    except tf.errors.NotFoundError:
        raise Exception("Failed to list files. Pattern (", os.path.join(data_dir, pattern), ") not matched. Check the directory exists and contains files.")

    training_filepaths_ds = full_filepaths_ds.take(training_file_count)

    test_filepaths_ds = full_filepaths_ds.skip(training_file_count)

    validation_filepaths_ds = test_filepaths_ds.skip(test_file_count)

    test_filepaths_ds = test_filepaths_ds.take(test_file_count)

    return training_filepaths_ds, validation_filepaths_ds, test_filepaths_ds, training_file_count, validation_file_count, test_file_count

def transform_dataset(dataset, transformation_function):
    return dataset.map(transformation_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def augment_dataset(dataset, transformation_function, augmented_dataset=None):
    if augmented_dataset is None:
        return dataset.concatenate(transform_dataset(dataset, transformation_function))

    return augmented_dataset.concatenate(transform_dataset(dataset, transformation_function))

def prepare_dataset(dataset, batch_size, cache=True, repeat=True, prefetch=True, shuffle=True, shuffle_seed=42, shuffle_buffer_size=1000):
    if (cache):
        if (isinstance(cache, str)):
            print("Opening cache or creating (%s)." % (cache))
            dataset = dataset.cache(cache)
        else:
            print("No cache path provided. Loading into memory.")
            dataset = dataset.cache()
    else:
        print("Not caching data. This may be slow.")

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

    dataset = dataset.repeat()

    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
