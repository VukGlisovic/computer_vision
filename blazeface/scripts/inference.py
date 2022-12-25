import sys
import logging
import argparse

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from decord import VideoReader, cpu
from tqdm import tqdm

from blazeface.constants import *
from blazeface.dataset import input_dataset, utils, prediction_decoder
from blazeface.model.blazeface import load_model


matplotlib.use('Agg')

logformat = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO, stream=sys.stdout)


def read_video(path, n_frames):
    """Reads a video from disk and yields its frames one by one.

    You can download example videos from:
    https://www.pexels.com/search/videos/selfie/

    Args:
        path (str):
        n_frames (int):

    Yields:
        np.ndarray
    """
    vr = VideoReader(path, ctx=cpu(0))
    logging.info('Number of frames in video: %s', len(vr))
    logging.info('Frames per second in video: %s', int(vr.get_avg_fps()))
    if n_frames < 0:
        n_frames = len(vr)
    frame_idxs = np.linspace(0, len(vr) - 1, n_frames).round().astype(int)
    for i in frame_idxs:
        frame = vr.get_batch([i])
        yield frame


def visualize_frame(frame, bboxes, lmarks):
    """Plots an image and adds the face bounding box rectangles together
    with the facial landmarks. Finally, converts the plot to a numpy array.

    Args:
        frame (np.ndarray):
        bboxes (tf.Tensor):
        lmarks (tf.Tensor):

    Returns:
        np.ndarray
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot image itself
    ax.imshow(frame)
    # plot bounding boxes
    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        rect = Rectangle((x1, y1), width, height, fc="None", ec='green', lw=2, alpha=0.7)
        ax.add_patch(rect)
    # plot landmarks
    for index, landmark in enumerate(lmarks):
        if tf.reduce_max(landmark) <= 0:
            continue
        ax.scatter(landmark[:, 0], landmark[:, 1], alpha=0.9, s=20)
    # convert plot to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img_array


def inference_on_frame(frame, model, all_anchors):
    """Runs inference on one frame and visualizes the bounding
    boxes and landmarks.

    Args:
        frame (np.ndarray):
        model (tf.keras.models.Model):
        all_anchors (tf.Tensor):

    Returns:
        np.ndarray
    """
    frame = frame.asnumpy()[0]
    # sneaky shortcut to crop the particular video I'm using
    frame = tf.image.crop_to_bounding_box(frame, 400, 120, target_height=1920, target_width=1920)
    # prepare frame for inference
    height, width, _ = frame.shape
    frame_processed = input_dataset.preprocess_image(frame)
    # run inference
    predictions = model.predict(tf.expand_dims(frame_processed, axis=0), verbose=0)
    # decode bounding box and landmark predictions
    pred_coordinates = prediction_decoder.get_bboxes_and_landmarks_from_deltas(all_anchors, predictions['deltas'])
    pred_scores = tf.cast(predictions['labels'], tf.float32)
    weighted_suppressed_data = prediction_decoder.weighted_suppression(pred_scores[0], pred_coordinates[0])
    weighted_bboxes = weighted_suppressed_data[..., 0:4]
    weighted_landmarks = weighted_suppressed_data[..., 4:]
    weighted_landmarks = tf.reshape(weighted_landmarks, (-1, N_LANDMARKS, 2))
    denormalized_bboxes = utils.denormalize_bboxes(weighted_bboxes, width=width, height=height)
    denormalized_landmarks = utils.denormalize_landmarks(weighted_landmarks, width=width, height=height)
    # plot detections on image
    img_array = visualize_frame(frame, denormalized_bboxes, denormalized_landmarks)
    return img_array


def main(video_path, checkpoint_path, n_frames):
    """Combines all functionality
    """
    model, all_anchors = load_model(checkpoint_path)
    frames = read_video(video_path, n_frames)  # frames iterator
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # be sure to use lower case
    video = cv2.VideoWriter('../data/inference/output_video.mp4', fourcc, 25, (1000, 1000))
    for frame in tqdm(frames):
        img = inference_on_frame(frame, model, all_anchors)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # opencv writes frames in videos in BGR channel order
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--video_path', type=str, required=True, help='Where to find the video to run inference on.')
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='Path to model checkpoint (hdf5 file).')
    parser.add_argument('-n', '--n_frames', type=int, default=-1, help='Number of frames to process. Default is -1 which means all frames.')
    known_args, _ = parser.parse_known_args()
    main(
        known_args.video_path,
        known_args.checkpoint_path,
        known_args.n_frames
    )
