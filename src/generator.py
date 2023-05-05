import os
import cv2
import random
import numpy as np

from src.utils.augmentation import flip_and_rotate

np.seterr(divide='ignore', invalid='ignore')


def collect_data(path, output_size):
    # Get image directories.
    _, folders, _ = os.walk(path).__next__()

    objects, object_names = {}, {}
    for i, name in enumerate(folders):
        # Get image files from current data folder.
        tmp = os.path.join(path, name)
        _, _, files = os.walk(tmp).__next__()

        # Make new key.
        objects[name] = []
        object_names[name] = []

        # Resize all images.
        for j, file in enumerate(files):
            # Skip non-image files.
            if file.split(".")[-1].lower() not in ["png", "jpg", "jpeg"]:
                continue

            # Open image.
            img = os.path.join(tmp, file)
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize.
            img_rsz = cv2.resize(img, output_size)

            # Normalize pixels.
            object_rsz = np.array(img_rsz) / 255.

            # Add to dictionary.
            objects[name].append(object_rsz)
            object_names[name].append(file)

            # --- display progression
            to_display = f"processed: folder {i+1}/{len(folders)} - file {j+1}/{len(files)}"
            print(to_display, end='\r')

        # Stack.
        objects[name] = np.stack(objects[name])

    return objects, object_names


class generator:
    def __init__(self, data, validation_data):
        # General attributes.
        self.epochs_done = 0
        self.index_in_epoch = 0

        # Get filter keys.
        self.train_data = data
        self.validation_data = validation_data
        self.valid_counter = 0

        # Shuffle indices.
        self.valid_indices = list(range(len(self.validation_data)))
        random.shuffle(self.valid_indices)

        self.train_indices = list(range(len(self.train_data)))
        random.shuffle(self.train_indices)

    def next_batch(self, batch_size, augment=True):
        # Define the start:end range of data to flush out for
        # training given the batch size.
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # Re-initialize and randomize training samples after every
        # epoch and continue flushing out batch data repeatedly.
        if self.index_in_epoch > len(self.train_indices) - 1:
            start = 0
            self.epochs_done += 1
            self.index_in_epoch = batch_size
            np.random.shuffle(self.train_indices)
        end = self.index_in_epoch

        data_batch = []
        for i in range(start, end):
            # Get data from index..
            index = self.train_indices[i]
            data = self.train_data[index]

            if augment:
                # Augment image.
                selects1 = [random.choice(range(3))]
                data = flip_and_rotate(
                    array=data,
                    select=selects1,
                    angle=random.choice(np.arange(0, 360, 90)))

            data_batch.append(data)

        return np.stack(data_batch)

    def next_validation(self):
        # Reset counter and randomize indices.
        if self.valid_counter > (len(self.validation_data) - 1):
            np.random.shuffle(self.valid_indices)
            self.valid_counter = 0

        # Get validation data index.
        index = self.valid_indices[self.valid_counter]
        self.valid_counter += 1

        # Return validation data at index.
        data = self.validation_data[index]
        return np.expand_dims(data, axis=0)
