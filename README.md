# ComputerVisionProject
In this project, we utilized DeepMind's TAPIR model to analyze videos of Parkinson's patients walking through a corridor while wearing eye-tracking glasses. Our goal was to map the patients' gaze positions onto a single heatmap.

Using TAPIR, we tracked any point on the screen with high accuracy. Leveraging computer vision tools, we processed the gaze data and successfully generated a comprehensive heatmap, providing valuable insights into the patients' visual attention patterns.

Before starting the analysis make sure to install all libraries using the requirements file.
The first step in the code is to provide the patient ID and select three file paths:
1.	Select the patient ID from a dropdown list.
2.	Choose the world view video file.
3.	Choose the world view video with gaze position marks (from the exports folder).
4.	Choose the gaze positions file (from the exports folder).
This first step can be done by running the cell: Load the file dialog
Afterward, import the relevant libraries, TAPIR model and run the functions to create the heatmap and dynamic heatmap.
Functions explanations:
def inference(frames, query_points):
    """Inference on one video."""

def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points."""

def FindHeatMap(status, group, id, start_timeSec, end_timeSec, world_path,world_circle_path, gaze_pos_path, second_halfQuery=None, second_halfQueryEnd=None, dynamic=False):
    """This function is responsible for generating heatmaps based on gaze data extracted from a video file."""

def pred_save(frames, query_points, width, height, patient_info, start_timeSec, end_timeSec, first_frame, status, group):
    """The pred_save function processes video frames and gaze data to generate and save heatmaps and gaze tracking data for a specific video segment."""

def get_heatmaps(width, height, patient_info, start_timeSec, end_timeSec, status, group, is_there_reading):
    """This function generates dynamic heatmaps based on gaze data for each frame of a video segment. The heatmaps visualize
    where the gaze points are concentrated over time, using Gaussian smoothing to highlight areas of interest."""

def save_video(frames, filename, fps=5):
    """This function saves a list of video frames as an output video file."""

def save_combined_video(video_path2, start_time_sec, output_filename, fps=5):
    """This function combines two video files side by side and saves the combined video."""




Next, upload the segments file (via the start_process function), which contains data for all patients (PD and OA) and their walking segments. Each segment represents straight walking in a corridor, with no turns.
# Get segments times split
def start_process():
    sheet = selected_patient                                                # From segments CSV file, select the patient's sheet
    segments = pd.read_excel(segments_file_path, sheet_name=sheet)  
    return segments, sheet

segments, sheet = start_process()                                           # Get the segments info, and the patients name

At this point, you can extract heatmaps. Several options are available: you can either generate separate heatmaps for different halves of a segment or any custom time range within the video. Another option is to process the second half of the segment and append it to the first half. This involves creating a heatmap for the first frame of the second half and appending it as gaze points to the last frame of the first half, then analyzing the segment as a whole. This method often produces better results than analyzing the entire segment at once.
Example:
# Process second half of video
segment_num = 2                                                             # Choose the segment to analyze
try:
    group = sheet[:2]; id = sheet[:10];  status = sheet[11:]
    times = segments[segments['SEGMENT'] == segment_num]['time_sec']        # Get start and end time of segment    
    start = times.iloc[0]; end = times.iloc[1]; half = (start + end) / 2    # Define the star, half and end times of walking segment
    # Process the segment's relevant times
    start_analyze = half; end_analyze = end-4                               # Define times to analyze
    FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, second_halfQuery=None)

except Exception as e:
    print(e)
# Load the second half processed data after it was analyzed in the last cell
if status:
    patient_info = f'{group}/{id}/{status}'
else:
    patient_info = f'{group}/{id}'

directory = os.path.join("Heatmap", patient_info, f"seg{start_analyze}_{end_analyze}",'gaze-data.csv' )
query_SecondHalf = pd.read_csv(directory)
query_SecondHalf = np.array(query_SecondHalf)

# Attach the second half to the first half and process as whole
try:
    group = sheet[:2]; id = sheet[:10];  status = sheet[11:]
    times = segments[segments['SEGMENT'] == segment_num]['time_sec']
    start = times.iloc[0]; end = times.iloc[1]; half = (start + end) / 2
    start_analyze = start+5; end_analyze = half                                # Define times to analyze
    FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, second_halfQuery=query_SecondHalf, second_halfQueryEnd=end)

except Exception as e:
    print(e)

If you want to create a dynamic heatmap, which displays the heatmap building alongside the world view video, take the processed video (from combining the two halves) and pass it into the FindHeatMap function with dynamic=True. This generates a heatmap for every frame of the processed video. Afterward, merge the new heatmaps with the original world view video.
Example:
# Dynamic
segment_num = 2                                                             # Choose the segment to analyze
try:
    group = sheet[:2]; id = sheet[:10];  status = sheet[11:]
    times = segments[segments['SEGMENT'] == segment_num]['time_sec']        # Get start and end time of segment    
    start = times.iloc[0]; end = times.iloc[1]; half = (start + end) / 2    # Define the star, half and end times of walking segment
    # Process the segment's relevant times
    start_analyze = start+5; end_analyze = end-4               
    heatmaps = FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, second_halfQuery=None, dynamic=True)
except Exception as e:
    print(e)

save_video(heatmaps, 'output_video.mp4', 5)

# Output file path
output_filename = filedialog.askdirectory(title="Where to drop the dynamic heat map?",
) + f'/DynamicHeatMap_{id}_{status}_{start_analyze}-{end_analyze}.mp4'

save_combined_video(world_circle_video_path, start_time_sec=start_analyze, output_filename=output_filename)


