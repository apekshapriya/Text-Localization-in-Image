
"""importing libraries"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob

from PIL import Image

import keras
from keras.applications.vgg16 import VGG16
import numpy as np
from keras import layers, Model, optimizers
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""
The dataset is created by following these steps.:
    Input images are RGB images consiting of text within it.
    The textual area in the input image is bounded by bounding boxes using an app called labelimg and has been manually tagged.
    Label images are formed using these tagged input image, making the text area masked as pixel-value 0 and 
    rest of the area as pixel-value 1. So the labelled images have pixel values 0 and 1.
"""


"""defining paths for input images and its corresponding target images(text area masked)"""

path_input_train = "../data/input_images/"
path_label_train = "../data/labeled_masked_images/"
train_input = glob.glob(os.path.join(path_input_train,"*"))


def preprocess_image(image):
    """
    reads the image in RGB mode and  resizes it into 224,224. Also normalized the image pixel in range (0-1)
    :param image:
    :return: image
    """
    img = Image.open(image).convert('RGB')
    img_crop = img.resize((224,224), Image.ANTIALIAS)
    img_crop = np.array(img_crop)
    scaled_img = np.divide(img_crop,255.0)
    return scaled_img


def create_train_test_set():
    """
    create the data-set. Stores the input image and its corresponding labelled images in two numpy arrays as input and
    label. Also splits the whole data-set into train and test.
    :return: train_input, train_label, test_input, test_label
    """
    train_input_images, train_label_images = [], []

    for image in train_input:
        image_name = image.split("/")[-1]
        if os.path.exists(os.path.join(path_label_train,image_name)):
            train_input_images.append(preprocess_image(image))
            train_label_images.append(preprocess_image(os.path.join(path_label_train,image_name)))

    train_input_images = np.array(train_input_images)
    train_label_images = np.array(train_label_images)
    test_input_images = train_input_images[-100:]
    test_label_images = train_label_images[-100:]
    train_input_images = train_input_images[:-100]
    train_label_images = train_label_images[:-100]

    print("train_images_length",len(train_input_images),len(train_label_images))
    
    print("test_images_length",len(test_input_images),len(test_label_images))
    return train_input_images, train_label_images, test_input_images, test_label_images


def define_cascaded_architecture():
    """
    Defines the model architecture. Takes the block 4's output from the standard vgg-16 network and uses this as
    input in further architecture. The following layers is used in the whole architecture:
    1) block4_pool output of vgg16-net
    2) (3,3), (3,7) and (7,3) convolution is applied on 1 parallely.
    3) The output of above three convolution is added and forms the next layer
    4) (2,2) pooling on 4
    5) (1,1) convolution on 4
    6) (1,1) convolution on 5
    7) up-sampling layer(deconvolution) so as to retain the original image size.
        (6 up-sampling layers has been used sequentially. The last up-sampling layer has activation sigmoid to get
        pixel from 0-1)

    :return: keras model
    """
    
    model = VGG16(weights='imagenet', include_top=False)
    block4 = model.get_layer(name='block4_pool').output
    partial_vgg = Model(model.input, block4)
    conv_tensor1_padded = keras.layers.ZeroPadding2D((1, 1),
                                        name='conv15_1_zp')(partial_vgg.output)

    conv_tensor_1 = keras.layers.Conv2D(512, (3, 3),
                                        activation='relu',
                                        name='conv_15_1')(conv_tensor1_padded)

    conv_tensor2_padded = keras.layers.ZeroPadding2D((1, 3),
                                        name='conv15_2_zp')(partial_vgg.output)

    conv_tensor_2 = keras.layers.Conv2D(512, (3, 7),
                                        activation='relu',
                                        name='conv_15_2')(conv_tensor2_padded)

    conv_tensor3_padded = keras.layers.ZeroPadding2D((3, 1),
                                        name='conv15_3_zp')(partial_vgg.output)

    conv_tensor_3 =  keras.layers.Conv2D(512, (7, 3), 
                                         activation='relu', 
                                         name='conv_15_3')(conv_tensor3_padded)

    sigma = keras.layers.Add()([conv_tensor_1, conv_tensor_2, conv_tensor_3])
    sigma_pool = keras.layers.MaxPool2D((2,2))(sigma)

    fully_conv_layer1 = layers.Conv2D(512, (1,1), strides=(1, 1), padding='same')(sigma_pool)
    fully_conv_layer2 = layers.Conv2D(512, (1,1), strides=(1, 1), padding='same')(fully_conv_layer1)

    transposed_conv1 = keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same')(fully_conv_layer2)
    transposed_conv3 = keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same')(transposed_conv1)
    transposed_conv4 = keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same')(transposed_conv3)
    transposed_conv5 = keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same')(transposed_conv4)
    transposed_conv6 = keras.layers.Conv2DTranspose(3, (4,4), strides=(2, 2), padding='same')(transposed_conv5)
    transposed_conv7 = keras.layers.Conv2DTranspose(3, (1,1), strides=(1, 1),activation='sigmoid', padding='same')\
        (transposed_conv6)

    embedding_model = Model(inputs=partial_vgg.input, outputs=transposed_conv7)
    return embedding_model


def save_model_fig(embedding_model):
    """
    Takes the model architecture and shows it in a understandable form of image
    :param embedding_model:
    :return:
    """

    plot_model(embedding_model, to_file='model.png')


def define_optimizer():
    """
    defines the optimizer for gradient descent
    :return:
    """
    
    optimizer = optimizers.adam()
    return optimizer


def train_model(embedding_model,train_input_images, train_label_images, reload_model=True):
    """
    Reloads the saved checkpoint if saved in the training process before. If not so, start fresh and saves the model
    to reuse it(saving from repeated task).

    :param embedding_model:
    :param train_input_images:
    :param train_label_images:
    :param reload_model:
    :return: trained model
    """
    
    if reload_model:
        embedding_model = load_model("model.hd5")
        print embedding_model
    else:
        optimizer = define_optimizer()
        embedding_model.compile('adam', loss='binary_crossentropy')
    embedding_model.fit(train_input_images[:100], train_label_images[:100], epochs=10, batch_size=32, shuffle=True,
                        verbose=True, callbacks=[ModelCheckpoint(filepath="../working_models/model.hd5", monitor='f1',
                        verbose=0, save_best_only=False), TensorBoard(log_dir='../logs/',histogram_freq=0,
                        write_graph=True, write_images=True)], validation_split=0.20)

    return embedding_model


def predict(embedding_model, input_images):
    """
    takes the trained model and images as input and gives the predicted output
    :param embedding_model:
    :param input_images:
    :return:
    """
    
    result = embedding_model.predict(input_images)
    return result


def metrics(embedding_model,input_images, label_images):
    """
    returns the result of the trained model
    :param embedding_model:
    :param input_images:
    :param label_images:
    :return:
    """
    embedding_model.evaluate(input_images, label_images)


def text_detection():
    """
    defines the whole flow of the process
    1) create dataset
    2) defines cascaded architecture
    3) train the model
    :return:
    """
    train_input_images, train_label_images,test_input_images, test_label_images = create_train_test_set()
    embedding_model = define_cascaded_architecture()
    save_model_fig(embedding_model)
    train_model(embedding_model, train_input_images, train_label_images, reload_model=True)


if __name__=="__main__":
    text_detection()


