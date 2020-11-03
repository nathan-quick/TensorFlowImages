#!/usr/bin/env python3
"""! TensorFlow demonstration of training off of your own images. 
Most of this is from the following links :
- https://www.tensorflow.org/tutorials/load_data/images
- https://www.tensorflow.org/tutorials
@author Seth McNeill
@date 2020 October 28
@copyright MIT
"""


# Includes
import datetime  # used for start/end times
import argparse  # This gives better commandline argument functionality
import doctest   # used for testing the code from docstring examples
import numpy as np  # for numerical operations
import os           # for file/OS operations
import PIL          # Image manipulation library
import PIL.Image    # Specific part of PIL
import tensorflow as tf     # actual TensorFlow library
import tensorflow_datasets as tfds  # To get datasets
import pathlib  # for path manipulations
import pdb      # for debugging
from tensorflow.keras import layers     # for building models
import matplotlib.pyplot as plt     # for plotting

# Global Variables


# Functions
def download_flowers(data_dir):
    """! Downloads the flower photos dataset
    @param data_dir The directory to save the dataset in

    @returns data_dir as returned by tf.keras.utils.get_file
    """
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname=os.path.join(data_dir,'flower_photos'), 
                                   untar=True)
    print(f"get_file returned directory: {data_dir}")
    return data_dir

def count_images(data_dir, file_type='jpg'):
    """! Counts the files of file_type in data_dir
    @param data_dir The directory to count in
    @param file_type A string containing the extension of the files to count (jpg is default)

    @returns The number of matching files found
    """
    data_dir = pathlib.Path(data_dir)
    print(f"pathlib returned directory: {data_dir}")
    image_count = len(list(data_dir.glob(f'*/*.{file_type}')))
    print(f"Directory ({data_dir}) contains {image_count} images")
    return image_count


def get_file_list(data_dir, file_type='jpg'):
    """! Creates a list of the files of file_type in data_dir
    @param data_dir The directory to count in
    @param file_type A string containing the extension of the files to count (jpg is default)

    @returns A list of all the files of type file_type in data_dir
    """
    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob(f'*/*.{file_type}'))
    return img_list


def show_penny(data_dir, file_type):
    """! Show an image of a penny from the dataset
    
    @param data_dir A pathlib.Path to the parent directory of the penny folder
    @param a string containing the extension of the files of interest
    """
    img_list = list(data_dir.glob(f"{os.path.join('penny','*.')}{file_type}"))
    n_images = len(img_list)
    rand_n = np.random.randint(0,n_images)
    im = PIL.Image.open(str(img_list[rand_n]))
    im.show()


def create_ds(data_dir, val_split, subset, seed, img_height, img_width,
            batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset=subset,
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = ds.class_names
    print(f"Class names found were:\n\t{class_names}")
    return ds


def create_model(train_ds, val_ds, epochs=3):
    num_classes = len(train_ds.class_names)
    # setup autotuning to improve performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return model


def test_dir_images(img_dir, file_type, prob_model, classes):
    test_list = get_file_list(img_dir, file_type)
    n_imgs = len(test_list)
    misclassified = []  # list of misclassified images
    for f in test_list:
        # see https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
        im = PIL.Image.open(f).resize((180,180))
        true_class = f.parts[-2]
        # convert it into a format that will work
        im_array = np.asarray(im)   # this shouldn't need scaling here since the model has a scaling step
        prob_im = prob_model(tf.convert_to_tensor([im_array])).numpy()
        prob_class = classes[np.argmax(prob_im)]
        if true_class != prob_class:
            misclassified.append(f)
            # plot the probabilities with image and class names
            plt.subplots(figsize=(8,6))
            plt.subplot(211)
            plt.imshow(im_array)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'A {true_class} Classified as a {prob_class}', fontsize=28)
            plt.subplot(212)
            plt.bar(np.arange(1,len(classes)+1), prob_im[0])
            plt.xticks(ticks=np.arange(1,len(classes)+1),
                        labels=classes, fontsize=16)
    accuracy = (n_imgs - len(misclassified))/n_imgs
    return (accuracy, misclassified, n_imgs)

def main():
    """! Main function
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctest', action='store_true',
                        help='Pass this flag to run doctest on the script')
    parser.add_argument('--training', required=True,
            help='Directory where training images are located in folders by class')
    parser.add_argument('--epochs', required=True, type=int,
            help='Number of epochs to use for training')
    parser.add_argument('--testing',
            help='Directory where testing images are located in folders by class')
    parser.add_argument('--download', action='store_true',
            help='Have the script download the flowers dataset and save it in --training')
    parser.add_argument('--filetype', default='jpg',
            help='File extension for image files')
    start_time = datetime.datetime.now()  # save the script start time
    args = parser.parse_args()  # parse the arguments from the commandline

    if(args.doctest):
        doctest.testmod(verbose=True)  # run the tests in verbose mode

    print("-------------------")
    print(f"Working with TensorFlow version {tf.__version__}")
    if args.download:
        data_dir = download_flowers(args.training)
        print(f"Training set saved to: {data_dir}")
        count_images(data_dir)
    else:
        count_images(args.training, args.filetype)
        data_dir = pathlib.Path(args.training)
    
    # show_penny(data_dir, args.filetype)
    train_ds = create_ds(data_dir, 0.2, 'training', 123, 180, 180, 32)
    val_ds = create_ds(data_dir, 0.2, 'validation', 123, 180, 180, 32)

    model = create_model(train_ds, val_ds, args.epochs)
    # create probability model
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    if args.testing:
        # run this by hand so that it can switch to a camera later
        # get a list of the images in the test directory
        test_list = get_file_list(args.testing, args.filetype)
        # load an image from the test group
        # see https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
        im_num = 0
        im = PIL.Image.open(test_list[im_num]).resize((180,180))
        true_class = test_list[im_num].parts[-2]
        # convert it into a format that will work
        im_array = np.asarray(im)   # this shouldn't need scaling here since the model has a scaling step
        # run the probability model against that image
        # this was tricky and best enlightenment came from learning that a dimension
        # of None means that a tensor doesn't know how many items are in that 
        # dimension. The previous datasets were a list of images so dim looked like
        # (54, 180, 180, 3). Need to add a dimension to make this work so just enclosed
        # it in []
        prob_im = probability_model(tf.convert_to_tensor([im_array])).numpy()
        prob_class = train_ds.class_names[np.argmax(prob_im)]
        # plot the probabilities with image and class names
        plt.subplots(figsize=(8,6))
        plt.subplot(211)
        plt.imshow(im_array)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'A {true_class} Classified as a {prob_class}', fontsize=28)
        plt.subplot(212)
        plt.bar(np.arange(1,len(train_ds.class_names)+1), prob_im[0])
        plt.xticks(ticks=np.arange(1,len(train_ds.class_names)+1),
                    labels=train_ds.class_names, fontsize=16)
        # rebuild for all images in a directory/directory structure
        accuracy = test_dir_images(args.testing, args.filetype, 
                probability_model, train_ds.class_names)
        print(f"Final accuracy on test images: {accuracy[0]:.4f} " +
              f"({len(accuracy[1])} of {accuracy[2]} misclassified)")
        plt.show()
    
    end_time = datetime.datetime.now()    # save the script end time
    print(f'{__file__} took {end_time - start_time} s to complete')


# This runs if the file is run as a script vs included as a module
if __name__ == '__main__':
    main()