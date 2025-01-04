MODEL_TYPE = 'bootstapir' # 'tapir' or 'bootstapir'
import functools
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import os

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tapnet.utils import model_utils

import cv2
from matplotlib.backend_bases import MouseButton
import pandas as pd
import copy
from scipy.spatial.distance import euclidean as eucli

from heatmap_func import *
from tqdm import tqdm
import time

# Load the pre-trained checkpoint
if MODEL_TYPE == 'tapir':
    checkpoint_path = 'tapnet/checkpoints/tapir_checkpoint_panning.npy'
else:
    checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.npy'

# Load the checkpoint state (parameters and model state)
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

# Set additional keyword arguments based on the model type
kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == 'bootstapir':
    kwargs.update(dict(
    pyramid_level=1,
    extra_convs=True,
    softmax_temperature=10.0))

# Initialize the TAPIR model
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)


# TAPIR Utility Functions

def inference(frames, query_points):
    """Inference on one video.

    Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = model_utils.preprocess_frames(frames)
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    outputs = tapir(video=frames, query_points=query_points, is_training=False, query_chunk_size=32)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

    # Binarize occlusions
    visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
    return tracks[0], visibles[0]

inference = jax.jit(inference)

def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points



resize_height = 128  # @param {type: "integer"}
resize_width = 128  # @param {type: "integer"}

def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points.

    Args:
    points: [num_points, 2], in [x, y]
    Returns:
    query_points: [num_points, 3], in [t, y, x]
    """
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points


def FindHeatMap(status, group, id, start_timeSec, end_timeSec, drive_path):
    # Define file paths
    base_path = drive_path + "/Studies/Fourth Year/Final Project/ET heatmap/data" 
    if status:
        patient_info = f'/{group}/{id}/{status}'
    else:
        patient_info = f'/{group}/{id}'
    world_path = base_path + patient_info + '/STRAIGHT/world.mp4'
    world_circle_path = base_path + patient_info + '/STRAIGHT/exports/000/world.mp4'
    gaze_pos_path = base_path + patient_info + '/STRAIGHT/exports/000/gaze_positions.csv'
    
    # Find frame indices corresponding to start and end times
    cap_world = cv2.VideoCapture(world_path)
    start_time = start_timeSec * 1000
    end_time = end_timeSec * 1000
    cap_world.set(cv2.CAP_PROP_POS_MSEC, start_time)
    ret, frame = cap_world.read()
    start_frame = int(cap_world.get(cv2.CAP_PROP_POS_FRAMES))
    cap_world.set(cv2.CAP_PROP_POS_MSEC, end_time)
    ret, frame = cap_world.read()
    end_frame = int(cap_world.get(cv2.CAP_PROP_POS_FRAMES))
    # end_frame = start_frame + frames_to_include
    height = frame.shape[0]; width = frame.shape[1]

    # Load gaze position data
    gaze_pos = pd.read_csv(gaze_pos_path)
    gaze_pos_conf = gaze_pos[gaze_pos['confidence'] > 0.6] # get only high confidence points
    gaze_x = gaze_pos_conf['norm_pos_x'] * width # manipulate normalized x to fit the frames
    gaze_y = (1 - gaze_pos_conf['norm_pos_y']) * height # manipulate normalized y to fit the frames
    world_idx = gaze_pos_conf['world_index'] 
    time_msec = 1000 * (gaze_pos['gaze_timestamp'] - gaze_pos['gaze_timestamp'][0])

    # Create a list to store frames within the range
    frames = []
    cap_world.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap_world.read()  # Read the frame
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    end_time = int(cap_world.get(cv2.CAP_PROP_POS_MSEC))
    cap_world.release()

    # Calculate frame differences for circular and world views
    cap = cv2.VideoCapture(world_circle_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    elapsed_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    lower_bound = elapsed_time - 10
    upper_bound = elapsed_time + 10
    # Find the indices where time_msec is within the range
    idx = np.where((time_msec > lower_bound) & (time_msec < upper_bound))[-1]
    frames_diff_circle = gaze_pos.loc[idx[0]]['world_index'] - start_frame 
    cap.release()
    cap = None
    
    # Frames difference calculation on world
    cap_world = cv2.VideoCapture(world_path)
    cap_world.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap_world.read()
    elapsed_time = cap_world.get(cv2.CAP_PROP_POS_MSEC)
    lower_bound = elapsed_time - 10
    upper_bound = elapsed_time + 10
    # Find the indices where time_msec is within the range
    idx = np.where((time_msec > lower_bound) & (time_msec < upper_bound))[-1]
    frames_diff_world = gaze_pos.loc[idx[0]]['world_index'] - start_frame 

    frames_diff_world_To_circle = frames_diff_circle - frames_diff_world

    cap_world.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap_world.read()
    cap_world.release() 
    cap_world = None
    video = media._VideoArray(frames)
    height, width = video.shape[1:3]
    
    # Predict Point Tracks for the Selected Points
    # Rescale video
    resize_height = 128
    resize_width = 128
    frames = media.resize_video(video, (resize_height, resize_width))

    # Define query points from the gaze point file
    query_points = np.array([]).reshape(0, 3)  # Initialize with 0 rows and 3 columns

    for select_frame in range(len(frames)):
        frame_num = select_frame + start_frame
        eye_reading = world_idx == frame_num + frames_diff_circle  - frames_diff_world_To_circle# get the relevant world indexes from gaze_pos
        if eye_reading.any():
            Xdot, Ydot = int(np.median(gaze_x[eye_reading])), int(np.median(gaze_y[eye_reading]))  # get the x,y positions from gaze_pos of the dot at the first frame
            select_points = np.array([[select_frame, Ydot, Xdot]])
        else:
            # No eye reading found; continue to the next frame
            # select_points = np.array([[select_frame, 0, 0]])
            continue
        query_points = np.append(query_points, select_points, axis=0)
    # Convert query points to the appropriate coordinate system
    query_points = transforms.convert_grid_coordinates(
        query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')
    print(f'Number of query points / Frames in segment: {len(query_points)} / {end_frame - start_frame}')
    pred_save(frames, query_points, width, height, patient_info, base_path, start_timeSec, end_timeSec, first_frame, status, group)
    
    
def pred_save(frames, query_points, width, height, patient_info, base_path, start_timeSec, end_timeSec, first_frame, status, group):
    resize_height = 128
    resize_width = 128
    # Predict points using the TAPIR model
    start_time = time.time()
    tracks, visibles = inference(frames, query_points)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Finished Tapir in {elapsed_time:.2f} seconds')
    

    # colormap = viz_utils.get_colors(101)
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))

    # Extract visibility information
    vis = visibles[:,0] 
    Xdot = tracks[:,0,0]; Ydot = tracks[:,0,1]
    
    # Create output directory
    directory = base_path + f"/Heatmap{patient_info}" + f"/seg{start_timeSec}_{end_timeSec}"
    if not os.path.exists(directory):        # Create the directory if it doesn't exist
        os.makedirs(directory)
        
    # Save gaze data and visibility information
    df = pd.DataFrame({'X': Xdot, 'Y': Ydot})
    df.to_csv(directory + f"/gaze-data.csv", index=False, header=False)

    df = pd.DataFrame({'Visibles': vis})
    df.to_csv(directory + f"/visibles-data.csv", index=False, header=False)
    # cv2.putText(first_frame, f'Times: {start_time/1000} - {end_time/1000} [s]', (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, thickness=4, color=(255, 0, 0)) 
    cv2.imwrite(directory + f"/CleanFirstFrame.png", first_frame)      # Save the first frame with additional text
    
    # Save and create Heatmap
    # Heatmap
    if group == 'EC':
        group = 'OA'
    text = f'Group: {group}, Status: {status}'
    # Define the font and scale
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3; thickness = 5; color = (0, 0, 255); text_position = (10, 100)
    cv2.putText(first_frame, text, text_position, font, font_scale, color, thickness)
    cv2.imwrite(directory + "/text_img.png", first_frame)

    input_path = directory + "/gaze-data.csv"
    display_width = int(width)
    display_height = int(height)
    alpha = 0.5
    # start_timeSec = str.replace(str(start_timeSec),'.', '_')
    output_name = directory
    output_name = output_name + f'/output'
    background_image = directory + "/text_img.png"
    ngaussian = 100
    sd = 5

    with open(input_path) as f:
        reader = csv.reader(f)
        raw = list(reader)
        
        gaza_data = []
        if len(raw[0]) == 2:
                gaze_data = list(map(lambda q: (int(float(q[0])), int(float(q[1])), 1), raw))
        else:
            gaze_data =  list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))
            
        draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name, imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)