# Project Overview

This project utilizes DeepMind's TAPIR model to analyze videos of Parkinson's patients walking through a corridor while wearing eye-tracking glasses. The primary goal is to map the patients' gaze positions into a single heatmap. Using TAPIR, we tracked any point on the screen, and with the help of computer vision tools, we generated both static and dynamic heatmaps to visualize the gaze patterns effectively.

Some files are not uploaded due to the sensitivity of medical information.

---

## Getting Started

### Prerequisites

Before starting the analysis, make sure to install all required libraries using the `requirements.txt` file.
Additionally, ensure that you clone and install the TAPIR repository from DeepMind. Follow the setup instructions in their repository to integrate it into this project.
https://github.com/google-deepmind/tapnet

### Initial Setup

1. **Provide the patient ID and select three file paths:**

   - Select the patient ID from a dropdown list.
   - Choose the world view video file.
   - Choose the world view video with gaze position marks (from the exports folder).
   - Choose the gaze positions file (from the exports folder).

   This setup can be done by running the `Load the file dialog` cell.

2. **Import Libraries and Models:**

   - Import all relevant libraries.
   - Load the TAPIR model.

3. **Run Functions:**

   - Execute the provided functions to create heatmaps and dynamic heatmaps.

---

## Functions Overview

### `inference(frames, query_points)`

Performs inference on a video.

### `convert_select_points_to_query_points(frame, points)`

Converts selected points to query points.

### `FindHeatMap(status, group, id, start_timeSec, end_timeSec, world_path, world_circle_path, gaze_pos_path, second_halfQuery=None, second_halfQueryEnd=None, dynamic=False)`

Generates heatmaps based on gaze data extracted from a video file.

### `pred_save(frames, query_points, width, height, patient_info, start_timeSec, end_timeSec, first_frame, status, group)`

Processes video frames and gaze data to generate and save heatmaps and gaze tracking data for a specific video segment.

### `get_heatmaps(width, height, patient_info, start_timeSec, end_timeSec, status, group, is_there_reading)`

Generates dynamic heatmaps for each frame of a video segment, highlighting gaze concentration areas using Gaussian smoothing.

### `save_video(frames, filename, fps=5)`

Saves a list of video frames as an output video file.

### `save_combined_video(video_path2, start_time_sec, output_filename, fps=5)`

Combines two video files side by side and saves the resulting video.

---

## Step-by-Step Process

### Upload the Segments File

1. Use the `start_process` function to upload the segments file containing data for all patients and their walking segments.

   ```python
   def start_process():
       sheet = selected_patient
       segments = pd.read_excel(segments_file_path, sheet_name=sheet)
       return segments, sheet

   segments, sheet = start_process()
   ```

2. Each segment represents straight walking in a corridor, with no turns.

### Extract Heatmaps

- **Generate separate heatmaps for halves or custom time ranges:**

  ```python
  segment_num = 2
  try:
      group = sheet[:2]; id = sheet[:10]; status = sheet[11:]
      times = segments[segments['SEGMENT'] == segment_num]['time_sec']
      start = times.iloc[0]; end = times.iloc[1]; half = (start + end) / 2

      start_analyze = half; end_analyze = end-4
      FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, second_halfQuery=None)
  except Exception as e:
      print(e)
  ```

- **Process the second half and append to the first half:**

  ```python
  directory = os.path.join("Heatmap", patient_info, f"seg{start_analyze}_{end_analyze}", 'gaze-data.csv')
  query_SecondHalf = pd.read_csv(directory).to_numpy()

  try:
      start_analyze = start+5; end_analyze = half
      FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, second_halfQuery=query_SecondHalf, second_halfQueryEnd=end)
  except Exception as e:
      print(e)
  ```

### Create Dynamic Heatmaps

To create a dynamic heatmap and merge it with the original world view video:

```python
try:
    start_analyze = start+5; end_analyze = end-4
    heatmaps = FindHeatMap(status, group, id, start_analyze, end_analyze, world_video_path, world_circle_video_path, gaze_pos_path, dynamic=True)
    save_video(heatmaps, 'output_video.mp4', 5)

    output_filename = filedialog.askdirectory(title="Where to drop the dynamic heat map?") + f'/DynamicHeatMap_{id}_{status}_{start_analyze}-{end_analyze}.mp4'
    save_combined_video(world_circle_video_path, start_time_sec=start_analyze, output_filename=output_filename)
except Exception as e:
    print(e)
```

---

## Notes

- Ensure all file paths are correctly selected before starting the analysis.
- Dynamic heatmaps provide a frame-by-frame visualization, making them ideal for detailed analysis.



![Video Example](https://github.com/NadavHolzerCode/FinalProject-ML/blob/main/GitGIF.gif)

