import torch
import gc
import os
from os import path
from argparse import ArgumentParser
import shutil
import requests
import urllib.request
import json
import pika

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

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials

from deeplabutils import DeeplabV3Plus
from deeplabutils import read_single_img
from deeplabutils import infer
from deeplabutils import decode_segmentation_masks


def setup_rabbitmq_parameters(username, password, host, port, virtual_host):
    credentials = pika.PlainCredentials(username, password)
    return pika.ConnectionParameters(host=host, port=port, credentials=credentials)

def create_rabbitmq_channel(parameters):
    connection = pika.BlockingConnection(parameters)
    return connection.channel()



credentials_dict = {
    "type": "service_account",
    "project_id": "single-verve-376219",
    "private_key_id": "0861a030b2cda0feb576636cefe23744823d1baf",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCtb9s0iz9nIJ27\n7Z5Od06yB538MqCSs9PkfpxH/VG3jQbBz4tX5RayAbKmLxi3dYKVfnMKgXteC1e9\nxR75zOCVnr+zf7TVJb2w7eI+/yT1Sly6HwKJLZceEOHlGffXJ4/+ILSBCced3F8L\nahc9JS99Crz5tOHeVkRI7l068p3q3K3PRQ4t73VziptjJ8txNqffStipRXwkszBp\nncNa52bI5l/RUtRheIpa3fM0A2UNHmVp1MU2gl7/n0MHUgp4PqyDZh6flKTVjO8E\nMQfFfg/Tc/4F8XQTZdbtv0l4jLbnffIf0fmoj6Y8mnCXZQG9ml9Amrm6VJam8qF7\n6P6Byo8nAgMBAAECggEAFQ8HYT5lIOr7bAWimqlu4zv8iVJGX/m4yT48UJtoEC7t\n+ptuMmptEokVPYtrZ187z1YOtuBY7+bVrQOhyrf/LvubEDr55IWUHkcMGRUW0jfI\nwYqhXrGr7ykinJRGHRg/Kh6jfCWJWNgYrESh9LkupnKm1nUJldsIqIhUxqMN3KXi\nqg2g7r8AbRKKaHyocum95Jccxx/oZu/Gw1VcBoPTX6KEykPDEfdacrTA5WdBInS+\nGa/lupgHA4ETRfp7Mp/t6gKUBW7Nh/DSX4TawJR8uboEf0VFNYshK/7b8NKtWGTG\n49DYcYIAk6g0jgGh4J7UAZJOzkwkNcDq48l7S8r+NQKBgQDgMALfhjLKaKUu+Ouy\nMgpsaE/POb7HEsfaUYiGOSpb0Fa0JxtSll147rDHvjp5YFimlS/cZaf7ShkwFNsv\nBpwLkKVOHUMBiICGD6OVJMdKnK5r3hiJlW2seKsjkxtQjGH0S0auTYwqqKJs3c1J\nDYZ13ccauV0GX11Ohe7XlM/ZfQKBgQDGDD5/xzfkrKLxZH140wIP/b6PTqhbcmTj\nPYrDf2SLVYnh2QH63HsBJGWLxlYBQTqi+Q/My5W7ukg7i4qWoHz0j/j//jLZ+9JY\n9JPPErARqUa629GhQs/x/k1+pmJawYme8G3AzNYFw1TWPp0EbRNDoPBF84mSLltx\nqVFpuEIMcwKBgQCqMaisurtqUEE+xLhiQn0JSbN1FViQ1uAkDIvBojpXE3YPNDUY\n4JA7k7FfIjpQFOWYKV/5SK9bJSi0CNFRBQqH+RqVj79jtZYksFC2lAI70XDU8Pnd\n0TQ+oCkES9SLtNdUV6VkA/kqFXWhgk0rbXorlt9lmV1WziUOzLzCqvWUHQKBgB+N\nSdO/oGb9HgSJNvgt3dFAYsCgDnBrPCl734Sf4hvUp9/kW81knPAkpUzsbz1J8BaQ\nyXSeJp++4M0jwROYQ/AOk+Ps0psp5GwpovbFiml153/Tj4U6iLiMBDqeNWMyHEPH\nGCU0PRCz+usbFJbk7cHDfSQX1Z4FZqooCIFoSpWDAoGBAI5kMSq233s2LlSuHg/5\nE6ejBm8DohaqpcivDg25lQEzyKe/99sgadKMjyAOcn6PUzOoiYzRQEnTmBQHasoC\ntJ9yyvsz5Hd/N6XWPzS6QbpzWa3n5Tz7B1B2eRD8ja8Xnmr89xl+7AyvgjDsAqWC\nMp87Rp+cGWxho2ZbF5jA8fm/\n-----END PRIVATE KEY-----\n",
    "client_email": "831716292954-compute@developer.gserviceaccount.com",
    "client_id": "113749115267900798731",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/831716292954-compute%40developer.gserviceaccount.com"
}
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    credentials_dict
)
client = storage.Client(credentials=credentials, project='single-verve-376219')
bucket = client.get_bucket('team-seven-bucket')

if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

device = 'cpu'

# set the amount of GPU memory to be used
# torch.cuda.set_per_process_memory_fraction(0.9)
# torch.backends.cuda.max_split_size_mb = 256

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

network = XMem(config, './saves/XMem.pth').eval().to(device)

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

# video_name = './video.mp4'
# mask_name = './first_frame.png'
#
# mask = np.array(Image.open(mask_name))
# print(np.unique(mask))
# num_objects = len(np.unique(mask)) - 1

def real_callback():
    url = "https://storage.googleapis.com/team-seven-bucket/peoplestation.mp4"  # Replace with the actual URL of the video
    filename = "theinputagain.mp4"  # Replace with the desired name of the video file

    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
            print(f"Video saved as {filename}")
    else:
        print("Failed to download video")

    input_file_name = './' + filename
    output_file_name = 'fromscriptdownloadagain.avi'

    torch.cuda.empty_cache()

    #maybe get mask here, run thru video for one frame, do mask op
    cap = cv2.VideoCapture(input_file_name)
    current_frame_index = 0
    while (cap.isOpened()):
        # load frame-by-frame
        _, frame = cap.read()
        frame = frame[:IMAGE_SIZE, :IMAGE_SIZE, :]
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
            print('mask info:!')
            print(mask.shape)
            print(np.unique(mask))
            num_objects = len(np.unique(mask)) - 1
        current_frame_index += 1

    cv2.destroyAllWindows()


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to be used
    out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (IMAGE_SIZE, IMAGE_SIZE))  # Video file output name, codec, fps, and frame size

    torch.cuda.empty_cache()

    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1))  # consecutive labels
    cap = cv2.VideoCapture(input_file_name)

    # You can change these two numbers
    frames_to_propagate = 120
    visualize_every = 1

    current_frame_index = 0

    with torch.cuda.amp.autocast(enabled=True):
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            frame = frame[:IMAGE_SIZE, :IMAGE_SIZE, :]
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
                #display(Image.fromarray(visualization))
                # Write the frame to the video file (must be square so write middle)
                out.write(visualization)
            cv2.waitKey(10)
            current_frame_index += 1
    out.release()
    cv2.destroyAllWindows()

    print('writing to bucket')
    blob = bucket.blob(output_file_name)
    blob.upload_from_filename(output_file_name)

    print('deleting from local storage')
    os.remove(input_file_name)
    os.remove(output_file_name)


rabbitmq_params = setup_rabbitmq_parameters('seven', 'supersecret', '34.123.41.144', '5672', '')

channel = create_rabbitmq_channel(rabbitmq_params)
print('channel created')

# create a queue
channel.queue_declare(queue='test-queue', durable=True)
print('queue declared')


# define callback function
def callback(ch, method, properties, body):
    req = body.decode('utf-8')
    print("Received message:", body.decode('utf-8'))


# start consuming messages
channel.basic_consume(queue='test-queue', on_message_callback=callback, auto_ack=True)
print('Waiting for messages...')
channel.start_consuming()

# num_objects = None
# mask = None
# while (cap.isOpened()):
#     # load frame-by-frame
#     _, frame = cap.read()
#     if frame is None or current_frame_index > 0:
#         break
#
#     if current_frame_index == 0:
#         image_tensor = read_single_img(frame)
#         prediction_mask = infer(image_tensor=image_tensor, model=model)
#         mask = decode_segmentation_masks(prediction_mask, colormap, 20)
#         unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
#         colormap = {}
#         for i in range(len(unique_colors)):
#             colormap[tuple(unique_colors[i])] = i
#
#
#         # Replace each pixel in the image with its corresponding colormap value
#         for i in range(mask.shape[0]):
#             for j in range(mask.shape[1]):
#                 mask[i,j] = colormap[tuple(mask[i,j])]
#         flat_img = Image.fromarray(mask[:,:,0], mode='L')
#         mask = np.array(flat_img)
#         print(mask.shape)
#         print(np.unique(mask))
#         num_objects = len(np.unique(mask)) - 1
#     current_frame_index += 1
#
# out.release()
# cv2.destroyAllWindows()
