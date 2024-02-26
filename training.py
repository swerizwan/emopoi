from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Concatenate, Lambda, Dot, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import os
from TTLayersfile import TT_Layer
import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Concatenate, Lambda, Dot, Reshape
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import os
import keras.activations
from keras import regularizers
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, LSTM, \
    BatchNormalization, Flatten, TimeDistributed, GlobalAveragePooling2D, Activation, Add, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# Function to create image lists for training, testing, and validation sets.
# It scans the provided image directory, categorizes images, and assigns them to respective sets.
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Organizes images into training, testing, and validation sets based on the specified parameters.

    Args:
        image_dir (str): Path to the directory containing labeled image folders.
        testing_percentage (int): Percentage of images to use for the testing set.
        validation_percentage (int): Percentage of images to use for the validation set.

    Returns:
        collections.OrderedDict: An ordered dictionary containing information about organized image sets.

    Raises:
        tf.errors.NotFoundError: If the specified image directory does not exist.

    Notes:
        This function utilizes TensorFlow's gfile module and other libraries to create structured datasets
        for machine learning model training.

    """
    if not gfile.Exists(image_dir):
        tf.logging.error(f"Image directory '{image_dir}' not found.")
        return None

    result = collections.OrderedDict()
    sub_dirs = [
        os.path.join(image_dir, item)
        for item in gfile.ListDirectory(image_dir)
    ]
    sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))

    for sub_dir in sub_dirs:
        extensions = ['png', 'PNG']
        file_list = []
        dir_name = os.path.basename(sub_dir)

        # Skip the main image directory
        if dir_name == image_dir:
            continue

        tf.logging.info(f"Looking for images in '{dir_name}'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, f'*.{extension}')
            file_list.extend(gfile.Glob(file_glob))

        if not file_list:
            tf.logging.warning('No files found')
            continue

        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)

            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """
    Get the full path of an image based on label, index, and category.

    Args:
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        label_name (str): Label of the image.
        index (int): Index of the image.
        image_dir (str): Root directory containing labeled image folders.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').

    Returns:
        str: Full path to the specified image.

    Raises:
        tf.errors.FatalError: If the specified label or category does not exist in the image_lists.

    Notes:
        This function is part of a broader image processing pipeline and helps construct paths to images
        based on their label, index, and category.
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)

    label_lists = image_lists[label_name]

    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)

    category_list = label_lists[category]

    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)

    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """
    Get the full path of the bottleneck file based on label, index, category, and architecture.

    Args:
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        label_name (str): Label of the image.
        index (int): Index of the image.
        bottleneck_dir (str): Directory to store bottleneck files.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').
        architecture (str): Model architecture used to create the bottleneck.

    Returns:
        str: Full path to the specified bottleneck file.

    Notes:
        This function is part of a broader image processing pipeline and helps construct paths to bottleneck
        files based on label, index, category, and architecture.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """
    Create a TensorFlow graph for the specified model.

    Args:
        model_info (dict): Information about the model, including file names and tensor names.

    Returns:
        Tuple[tf.Graph, tf.Tensor, tf.Tensor]: TensorFlow graph, bottleneck tensor, and resized input tensor.

    Notes:
        This function is part of a broader image processing pipeline and is used to load a pre-trained model's
        graph for further processing.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """
    Run the bottleneck operation on a given image.

    Args:
        sess (tf.Session): TensorFlow session.
        image_data (bytes): Raw image data.
        image_data_tensor (tf.Tensor): TensorFlow tensor for raw image data.
        decoded_image_tensor (tf.Tensor): TensorFlow tensor for decoded image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.

    Returns:
        np.ndarray: Bottleneck values for the given image.

    Notes:
        This function is part of a broader image processing pipeline and is used to extract bottleneck
        features from a given image using a pre-trained model.
    """
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})

    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def maybe_download_and_extract(data_url):
    """
    Download and extract a file from the specified data URL.

    Args:
        data_url (str): URL of the data file.

    Notes:
        This function ensures that the file is downloaded and extracted to the specified destination directory.
    """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                        'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """
    Ensure that the specified directory exists; if not, create it.

    Args:
        dir_name (str): Name of the directory.

    Notes:
        This function checks if the specified directory exists, and if not, it creates the directory.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """
    Create a bottleneck file for the specified image.

    Args:
        bottleneck_path (str): Path to the bottleneck file.
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        label_name (str): Label of the image.
        index (int): Index of the image.
        image_dir (str): Root directory containing labeled image folders.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').
        sess (tf.Session): TensorFlow session.
        jpeg_data_tensor (tf.Tensor): TensorFlow tensor for JPEG image data.
        decoded_image_tensor (tf.Tensor): TensorFlow tensor for decoded image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.

    Raises:
        RuntimeError: If an error occurs during the processing of the image.

    Notes:
        This function is part of a broader image processing pipeline and creates bottleneck files for each image.
    """
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """
    Get or create a bottleneck file for the specified image.

    Args:
        sess (tf.Session): TensorFlow session.
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        label_name (str): Label of the image.
        index (int): Index of the image.
        image_dir (str): Root directory containing labeled image folders.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').
        bottleneck_dir (str): Directory to store bottleneck files.
        jpeg_data_tensor (tf.Tensor): TensorFlow tensor for JPEG image data.
        decoded_image_tensor (tf.Tensor): TensorFlow tensor for decoded image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.
        architecture (str): Model architecture used to create the bottleneck.

    Returns:
        np.ndarray: Bottleneck values for the specified image.

    Notes:
        This function is part of a broader image processing pipeline and either retrieves existing bottleneck
        values or creates a new bottleneck file and returns its values.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

'''
def f1(y_hat, y_true, model='multi'):
  epsilon = 1e-7
  y_hat = tf.round(y_hat)#将经过sigmoid激活的张量四舍五入变为0，1输出
  tp = tf.reduce_sum(tf.cast(y_hat*y_true, 'float'), axis=0)
  #tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
  fp = tf.reduce_sum(tf.cast(y_hat*(1-y_true), 'float'), axis=0)
  fn = tf.reduce_sum(tf.cast((1-y_hat)*y_true, 'float'), axis=0)
  p = tp/(tp+fp+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
  r = tp/(tp+fn+epsilon)
  f1 = 2*p*r/(p+r+epsilon)
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  if model == 'single':
    return f1
  if model == 'multi':
    return tf.reduce_mean(f1)
'''

def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """
    Cache bottlenecks for all images in the specified image lists.

    Args:
        sess (tf.Session): TensorFlow session.
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        image_dir (str): Root directory containing labeled image folders.
        bottleneck_dir (str): Directory to store bottleneck files.
        jpeg_data_tensor (tf.Tensor): TensorFlow tensor for JPEG image data.
        decoded_image_tensor (tf.Tensor): TensorFlow tensor for decoded image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.
        architecture (str): Model architecture used to create the bottleneck.

    Notes:
        This function iterates through all images in the image lists and caches their bottleneck values.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
    """
    Get a random sample of cached bottlenecks.

    Args:
        sess (tf.Session): TensorFlow session.
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        how_many (int): Number of bottlenecks to retrieve. If -1, retrieve all bottlenecks.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').
        bottleneck_dir (str): Directory storing bottleneck files.
        image_dir (str): Root directory containing labeled image folders.
        jpeg_data_tensor (tf.Tensor): TensorFlow tensor for JPEG image data.
        decoded_image_tensor (tf.Tensor): TensorFlow tensor for decoded image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.
        architecture (str): Model architecture used to create the bottleneck.

    Returns:
        tuple: A tuple containing lists of bottlenecks, ground truths, and filenames.

    Notes:
        This function retrieves a random sample of cached bottlenecks along with their ground truths and filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor, architecture)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
    """
    Get a random sample of distorted bottlenecks.

    Args:
        sess (tf.Session): TensorFlow session.
        image_lists (collections.OrderedDict): Dictionary containing information about image sets.
        how_many (int): Number of distorted bottlenecks to retrieve.
        category (str): Category of the image set (e.g., 'training', 'testing', 'validation').
        image_dir (str): Root directory containing labeled image folders.
        input_jpeg_tensor (tf.Tensor): TensorFlow tensor for JPEG image data.
        distorted_image (tf.Tensor): TensorFlow tensor for distorted image.
        resized_input_tensor (tf.Tensor): TensorFlow tensor for resized input.
        bottleneck_tensor (tf.Tensor): TensorFlow tensor for bottleneck values.

    Returns:
        tuple: A tuple containing lists of distorted bottlenecks and their ground truths.

    Notes:
        This function retrieves a random sample of distorted bottlenecks along with their ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()

        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck_values)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """
    Check if image distortion is needed.

    Args:
        flip_left_right (bool): Whether to randomly flip images horizontally.
        random_crop (int): Percentage of image to randomly crop.
        random_scale (int): Percentage of image to randomly scale.
        random_brightness (int): Percentage of image to randomly adjust brightness.

    Returns:
        bool: True if any distortion is needed, False otherwise.

    Notes:
        This function checks if image distortion is needed based on the specified parameters.
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """
    Add input distortions to the model.

    Args:
        flip_left_right (bool): Whether to randomly flip images horizontally.
        random_crop (int): Percentage of image to randomly crop.
        random_scale (int): Percentage of image to randomly scale.
        random_brightness (int): Percentage of image to randomly adjust brightness.
        input_width (int): Width of the input image.
        input_height (int): Height of the input image.
        input_depth (int): Depth (number of channels) of the input image.
        input_mean (float): Mean value for input normalization.
        input_std (float): Standard deviation for input normalization.

    Returns:
        tuple: A tuple containing TensorFlow placeholders for JPEG image data and distorted result.

    Notes:
        This function adds input distortions to the model based on the specified parameters.
    """
    jpeg_data = tf.placeholder(tf.string, name='DistortPNGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """
    Attach summaries to a TensorFlow variable.

    Args:
        var (tf.Variable): TensorFlow variable.

    Notes:
        This function attaches summaries for mean, standard deviation, maximum, minimum, and histogram to the variable.
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.models import Model

def normalization_relu_add(x):
    """
    Apply batch normalization followed by ReLU activation.

    Args:
    - x: Input tensor.

    Returns:
    - Tensor after normalization and ReLU activation.
    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def residual_block_add(x, filters_num_in, filters_num_out, id_block):
    """
    Create a residual block with skip connection for identity mapping.

    Args:
    - x: Input tensor.
    - filters_num_in: Number of filters for the intermediate convolution layers.
    - filters_num_out: Number of filters for the final convolution layer.
    - id_block: Identifier for the block.

    Returns:
    - Tensor after applying the residual block.
    """
    tensor_input = x
    x = Conv2D(filters_num_in, (1, 1), name=id_block + '_1')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_in, (3, 3), padding='same', name=id_block + '_2')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_out, (1, 1), name=id_block + '_3')(x)
    x = BatchNormalization()(x)
    x = Add()([tensor_input, x])
    x = Activation('relu')(x)
    return x

def convolutional_block_add(x, filters_num_in, filters_num_out, id_block):
    """
    Create a convolutional block with a skip connection for downsampling.

    Args:
    - x: Input tensor.
    - filters_num_in: Number of filters for the intermediate convolution layers.
    - filters_num_out: Number of filters for the final convolution layer.
    - id_block: Identifier for the block.

    Returns:
    - Tensor after applying the convolutional block.
    """
    tensor_input = x
    x = Conv2D(filters_num_in, (1, 1), strides=(2, 2), name=id_block + '_1')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_in, (3, 3), padding='same', name=id_block + '_2')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_out, (1, 1), name=id_block + '_3')(x)
    shortcut = Conv2D(filters_num_out, (1, 1), strides=(2, 2), name=id_block + '_shortcut')(tensor_input)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def netface(input_image, network_id=""):
    """
    Build a ResNet model for a single image.

    Args:
    - input_image: Input tensor for the image.
    - network_id: Identifier for the network.

    Returns:
    - Keras Model for the ResNet model.
    """
    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = normalization_relu_add(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = convolutional_block_add(x, 64, 256, '%sres1_1' % network_id)
    x = residual_block_add(x, 64, 256, '%sres1_2' % network_id)

    x = convolutional_block_add(x, 256, 512, '%sres4_1' % network_id)
    x = residual_block_add(x, 256, 512, '%sres4_2' % network_id)
    x = GlobalAveragePooling2D()(x)

    model = Model(input_image, x, name='%s_resnet' % network_id)
    return model

def time_distributed_netface(input_image_s, network_id=""):
    """
    Build a time-distributed ResNet model for sequences of images.

    Args:
    - input_image_s: Input tensor for the sequence of images.
    - network_id: Identifier for the network.

    Returns:
    - Keras Model for the time-distributed ResNet model.
    """
    return TimeDistributed(netface(Input(input_image_s), network_id))

def netlivestreaming(input_image, network_id=""):
    """
    Build a modified ResNet model for a single image.

    Args:
    - input_image: Input tensor for the image.
    - network_id: Identifier for the network.

    Returns:
    - Keras Model for the modified ResNet model.
    """
    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = normalization_relu_add(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = convolutional_block_add(x, 64, 128, '%sres1_1' % network_id)
    x = residual_block_add(x, 64, 128, '%sres1_2' % network_id)

    x = convolutional_block_add(x, 128, 256, '%sres2_1' % network_id)
    x = residual_block_add(x, 128, 256, '%sres2_2' % network_id)

    x = convolutional_block_add(x, 256, 512, '%sres4_1' % network_id)
    x = residual_block_add(x, 256, 512, '%sres4_2' % network_id)
    x = GlobalAveragePooling2D()(x)

    model = Model(input_image, x, name='%s_resnet' % network_id)
    return model

def time_distributed_netlivestreaming(input_image_s, network_id=""):
    """
    Build a time-distributed modified ResNet model for sequences of images.

    Args:
    - input_image_s: Input tensor for the sequence of images.
    - network_id: Identifier for the network.

    Returns:
    - Keras Model for the time-distributed modified ResNet model.
    """
    return TimeDistributed(netlivestreaming(Input(input_image_s), network_id))


def tt_fusion_model(time_steps, flag_model):
    """
    Build a fusion model combining modified ResNet models with Tensor Train (TT) layer.

    Args:
    - time_steps: Number of time steps in the sequence.
    - flag_model: Identifier for the model.

    Returns:
    - Keras Model for the fusion model.
    """
    input_face = Input((time_steps, 64, 64, 3))
    input_video = Input((time_steps, 128, 128, 3))
    bias = Input((1,))

    features_face = time_distributed_netface((64, 64, 3), "face_")(input_face)
    features_video = time_distributed_netlivestreaming((128, 128, 3), "game_")(input_video)

    features_face = BatchNormalization()(features_face)
    features_face = Dropout(0.2)(features_face)
    features_face = LSTM(128, return_sequences=True)(features_face)
    features_face = LSTM(128, return_sequences=False)(features_face)

    features_video = BatchNormalization()(features_video)
    features_video = Dropout(0.2)(features_video)
    features_video = LSTM(128, return_sequences=True)(features_video)
    features_video = LSTM(128, return_sequences=False)(features_video)

    reshape_1 = Reshape((1, 129))(Concatenate()([bias, features_face]))
    reshape_2 = Reshape((1, 129))(Concatenate()([bias, features_video]))

    x = Dot(axes=1)([reshape_1, reshape_2])
    x = Reshape((1, 129 * 129))(x)
    feats_hidden = Reshape((129, 129, 129))(x)

    print("Tensor Shape: ", feats_hidden.shape)
    feats_hidden = TT_Layer(list_shape_input=[3, 43, 129, 43, 3],
                            list_shape_output=[2, 4, 4, 4, 3],
                            list_ranks=[1, 2, 4, 4, 2, 1],
                            activation='relu', initializer_kernel=keras.regularizers.l2(5e-4), dtype=feats_hidden.dtype, debug=False)(feats_hidden)
    print("After TT Shape: ", feats_hidden.shape)

    feats_hidden = BatchNormalization()(feats_hidden)
    feats_hidden = Dropout(0.2)(feats_hidden)

    outputs = []

    feats_video = Dense(128, activation="relu")(feats_hidden)
    video = Dense(8, activation='softmax')(feats_video)
    outputs.append(video)

    model = Model([input_face, input_video], outputs)

    opt = Adam(lr=0.0005)
    plot_model(model, to_file='model_imgs/%s_tt.png' % flag_model, show_shapes=True)
    print(model.summary())
    return model

def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
    """
    Add the final training operations to the TensorFlow graph.

    Args:
    - class_count: Number of classes in the classification task.
    - final_tensor_name: Name for the final tensor in the graph.
    - bottleneck_tensor: Placeholder for the bottleneck feature.
    - bottleneck_tensor_size: Size of the bottleneck feature.

    Returns:
    - Tuple containing the training step, cross entropy mean, bottleneck input placeholder,
      ground truth input placeholder, and the final tensor.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    Add the evaluation step to the TensorFlow graph.

    Args:
    - result_tensor: The result tensor to be evaluated.
    - ground_truth_tensor: Placeholder for the ground truth labels.

    Returns:
    - Tuple containing the evaluation step and prediction tensor.
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction

def save_graph_to_file(sess, graph, graph_file_name):
    """
    Save the TensorFlow graph to a file.

    Args:
    - sess: TensorFlow session containing the graph.
    - graph: TensorFlow graph to be saved.
    - graph_file_name: Name of the file to save the graph.
    """
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return

def prepare_file_system():
    """
    Prepare the file system for storing summaries and intermediate output graphs.
    """
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return

def create_model_info(architecture):
    """
    Create model information based on the specified architecture.

    Args:
    - architecture: String specifying the architecture of the model.

    Returns:
    - Dictionary containing model information.
    """
    architecture = architecture.lower()
    if architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'", architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
            version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""", version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
            size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""", size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3], architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pb'
        else:
            model_base_name = 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """
    Add JPEG decoding operations to the TensorFlow graph.

    Args:
    - input_width: Width of the input image.
    - input_height: Height of the input image.
    - input_depth: Depth (number of channels) of the input image.
    - input_mean: Mean value for normalization.
    - input_std: Standard deviation value for normalization.

    Returns:
    - Tuple containing JPEG data placeholder and the preprocessed image tensor.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodePNGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image



def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare necessary directories  that can be used during training
  prepare_file_system()

  # Gather information about the model architecture we'll be using.
  model_info = create_model_info(FLAGS.architecture)
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Set up the pre-trained graph.
  maybe_download_and_extract(model_info['data_url'])
  graph, bottleneck_tensor, resized_image_tensor = (
      create_model_graph(model_info))

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')
    return -1

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, model_info['input_width'],
           model_info['input_height'], model_info['input_depth'],
           model_info['input_mean'], model_info['input_std'])
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, FLAGS.architecture)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             FLAGS.architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))

      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name)

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks)))

    '''
    with tf.Session() as sess:
      f1_score = f1(y_hat, y_true)
      print('F1 score:', sess.run(f1)
    '''
    
    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' %
                          (test_filename,
                           list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(sess, graph, FLAGS.output_graph)
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=6000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--architecture',
      type=str,
      default='MobileNet_1.0_224',
      help="""\
      Chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
