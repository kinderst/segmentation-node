import torch

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


def read_single_img(img_param):
    image = tf.convert_to_tensor(img_param)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap_param, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap_param[l, 0]
        g[idx] = colormap_param[l, 1]
        b[idx] = colormap_param[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

print('device count:')
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

network = torch.nn.DataParallel(XMem(config, './saves/XMem.pth').eval()).to(device)

# Loading the Colormap
colormap = loadmat(
    "./deeplab_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

IMAGE_SIZE = 512
NUM_CLASSES = 20

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

model.load_weights('./deeplabv3weights.h5')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to be used
out = cv2.VideoWriter('testvide.mp4', fourcc, 20.0, (512, 512))

torch.cuda.empty_cache()

#maybe get mask here, run thru video for one frame, do mask op
cap = cv2.VideoCapture('./kidthumb.mp4')
current_frame_index = 0

num_objects = None
mask = None
while (cap.isOpened()):
    # load frame-by-frame
    _, frame = cap.read()
    if frame is None or current_frame_index > 0:
        break

    if current_frame_index == 0:
        image_tensor = read_single_img(frame)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        mask = decode_segmentation_masks(prediction_mask, colormap, 20)
        unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        colormap = {}
        for i in range(len(unique_colors)):
            colormap[tuple(unique_colors[i])] = i


        # Replace each pixel in the image with its corresponding colormap value
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i,j] = colormap[tuple(mask[i,j])]
        flat_img = Image.fromarray(mask[:,:,0], mode='L')
        mask = np.array(flat_img)
        print(mask.shape)
        print(np.unique(mask))
        num_objects = len(np.unique(mask)) - 1
    current_frame_index += 1

out.release()
cv2.destroyAllWindows()


torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
cap = cv2.VideoCapture('./kidthumb.mp4')

# You can change these two numbers
frames_to_propagate = 30
visualize_every = 1

start_row = int((720 - 512) / 2)
end_row = start_row + 512
start_col = int((1280 - 512) / 2)
end_col = start_col + 512

current_frame_index = 0

with torch.cuda.amp.autocast(enabled=True):
    while (cap.isOpened()):
        # load frame-by-frame
        _, frame = cap.read()
        frame = frame[start_row:end_row, start_col:end_col,:]
        print(frame.shape)
        if frame is None or current_frame_index > frames_to_propagate:
            break

        # convert numpy array to pytorch tensor format
        frame_torch, _ = image_to_torch(frame, device=device)
        print(frame_torch.shape)
        if current_frame_index == 0:
            # initialize with the mask
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
            print(mask_torch.shape)
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch)

        # argmax, convert to numpy
        prediction = torch_prob_to_numpy_mask(prediction)

        if current_frame_index % visualize_every == 0:
            visualization = overlay_davis(frame, prediction)
            # Write the frame to the video file (must be square so write middle)
            out.write(visualization)

        current_frame_index += 1
out.release()
cv2.destroyAllWindows()
