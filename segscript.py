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

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from deeplabutils import DeeplabV3Plus
from deeplabutils import read_single_img
from deeplabutils import infer
from deeplabutils import decode_segmentation_masks

import gizeh
from moviepy.editor import ImageSequenceClip


def setup_rabbitmq_parameters(username, password, host, port, virtual_host):
    credentials_pika = pika.PlainCredentials(username, password)
    return pika.ConnectionParameters(host=host,
                                     port=port,
                                     credentials=credentials_pika)


def create_rabbitmq_channel(parameters):
    connection = pika.BlockingConnection(parameters)
    return connection.channel()


google_json = None
# Open the JSON file for reading
with open('./google-creds.json', 'r') as f:
    # Parse the JSON data from the file into a dictionary
    google_json = json.load(f)

google_credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    google_json
)
client = storage.Client(credentials=google_credentials, project='single-verve-376219')
bucket = client.get_bucket('team-seven-bucket')

# Use a service account.
cred = credentials.Certificate('./team-seven-fire-firebase-admin.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()

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

# # Loading the Colormap
# colormap = loadmat(
#     "./deeplab_colormap.mat"
# )["colormap"]
# colormap = colormap * 100
# colormap = colormap.astype(np.uint8)

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


def real_callback(ch, method, properties, body):
    req = body.decode('utf-8')
    print("Received message:", body.decode('utf-8'))
    req_data = json.loads(req)
    firebase_id, url, output_type, num_frames, fps, filename = None, None, None, None, None, None
    try:
        print(req_data["originalUrl"])
        firebase_id = req_data["firebaseId"]
        url = req_data["originalUrl"]  # Replace with the actual URL of the video
        output_type = req_data["outputType"]
        num_frames = req_data["numFrames"]
        fps = req_data["fps"]
        filename = firebase_id  # Replace with the desired name of the video file
    except:
        # Acknowledge the message
        print('bad message')
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    if not firebase_id or not url or not output_type or not num_frames or not fps or not filename:
        print('something missing')
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    doc_ref = db.collection(u'videos').document(u''+firebase_id)
    doc_ref.set({
        u'status': 'message received, beginning video download'
    }, merge=True)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
                print(f"Video saved as {filename}")
        else:
            print("Failed to download video")
            doc_ref.set({
                u'status': 'failed, video download error'
            }, merge=True)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
    except:
        print("bad video, exiting")
        doc_ref.set({
            u'status': 'failed, could not download video'
        }, merge=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    input_file_name = './' + filename
    output_file_name = 'output' + filename + '.' + output_type

    torch.cuda.empty_cache()

    #maybe get mask here, run thru video for one frame, do mask op
    cap = cv2.VideoCapture(input_file_name)
    current_frame_index = 0
    start_x, end_x, start_y, end_y = None, None, None, None
    while (cap.isOpened()):
        # load frame-by-frame
        _, frame = cap.read()
        middle_x = frame.shape[0] // 2
        middle_y = frame.shape[1] // 2
        start_x = middle_x - 256
        end_x = middle_x + 256

        start_y = middle_y - 256
        end_y = middle_y + 256
        frame = frame[start_x:end_x, start_y:end_y, :]
        if frame is None or current_frame_index > 0:
            break

        if current_frame_index == 0:
            image_tensor = read_single_img(frame)
            prediction_mask = infer(image_tensor=image_tensor, model=model)
            # Loading the Colormap
            colormap = loadmat(
                "./deeplab_colormap.mat"
            )["colormap"]
            colormap = colormap * 100
            colormap = colormap.astype(np.uint8)
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

    doc_ref.set({
        u'status': 'video downloaded successfully, and mask created, beginning segmentation'
    }, merge=True)

    fourcc = None
    if output_type == 'mp4':
        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to be used
        fourcc = 'libx264'
    elif output_type == 'avi':
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc = 'png'

    #out = cv2.VideoWriter(output_file_name, fourcc, fps, (IMAGE_SIZE, IMAGE_SIZE))  # Video file output name, codec, fps, and frame size

    torch.cuda.empty_cache()

    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1))  # consecutive labels
    cap = cv2.VideoCapture(input_file_name)

    # You can change these two numbers
    frames_to_propagate = num_frames
    visualize_every = 1

    current_frame_index = 0
    img_arr = []
    with torch.cuda.amp.autocast(enabled=True):
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            frame = frame[start_x:end_x, start_y:end_y, :]
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
                #out.write(visualization)
                img_arr.append(visualization)

            # every 10 frames, write to firestore the progress
            if current_frame_index % 10 == 0:
                doc_ref.set({
                    u'status': 'segmented ' + str(current_frame_index) + '/' + str(frames_to_propagate) + ' frames'
                }, merge=True)

            cv2.waitKey(10)
            current_frame_index += 1
    #out.release()
    cv2.destroyAllWindows()

    doc_ref.set({
        u'status': 'finished segmentation, writing to bucket and cleaning up'
    }, merge=True)

    clip = ImageSequenceClip(img_arr, fps=fps)
    clip.write_videofile(output_file_name, codec=fourcc)

    print('writing to bucket')
    blob = bucket.blob(output_file_name)
    blob.upload_from_filename(output_file_name)

    print('deleting from local storage')
    os.remove(input_file_name)
    os.remove(output_file_name)

    doc_ref.set({
        u'status': 'finished',
        u'outputUrl': 'https://storage.googleapis.com/team-seven-bucket/' + output_file_name
    }, merge=True)

    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)


rabbitmq_params = setup_rabbitmq_parameters('seven', 'supersecret', '34.123.41.144', '5672', '')

channel = create_rabbitmq_channel(rabbitmq_params)
print('channel created')

# create a queue
channel.queue_declare(queue='test-queue', durable=True)
print('queue declared')

channel.queue_purge(queue='test-queue')
print('cleared queue')


# define callback function
def callback(ch, method, properties, body):
    req = body.decode('utf-8')
    print("Received message:", body.decode('utf-8'))
    req_data = json.loads(req)
    print(req_data["originalUrl"])

try:
    # start consuming messages
    channel.basic_consume(queue='test-queue', on_message_callback=real_callback)
    print('Waiting for messages...')
    channel.start_consuming()
except pika.exceptions.ConnectionClosed:
    print("lost connection")

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
