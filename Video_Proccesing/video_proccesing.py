import cv2
import os
import shutil
import numpy as np
from moviepy.editor import VideoFileClip


def get_video_details(video_path) -> dict:
    clip = VideoFileClip(video_path)
    fps = clip.fps  # Gets the frames per second of the video
    duration = clip.duration  # Duration in seconds
    width, height = clip.size  # Width and height of the video
    audio_fps = clip.audio.fps if clip.audio else None  # Audio frames per second, if audio is present

    # Construct a dictionary with the video details
    video_details = {
        "frame_rate": fps,
        "duration_seconds": duration,
        "resolution": (width, height),
        "audio_frame_rate": audio_fps,
    }

    # Optionally, include more details based on your requirements
    return video_details


def is_same_frame(frame1, frame2) -> bool:
    if frame1 is None or frame2 is None:
        return False
    # Comparing the two frames for equality
    difference = cv2.absdiff(frame1, frame2)
    return not np.any(difference)


def identify_format_and_split_frames(video_path):
    # Extract the video name without extension and format
    video_name = os.path.basename(video_path).split('.')[0]
    video_format = video_path.split('.')[-1]
    target_dir = os.path.join(os.path.dirname(video_path), video_name)  # Directory to save frames

    # Check if target_dir exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # Remove the directory if it exists
    os.makedirs(target_dir)  # Create the directory

    print(f"Video format identified as: {video_format}")

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)

    last_frame = None
    saved_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 320x320
        resized_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)

        # Skip saving if the current frame is identical to the last saved frame
        if not is_same_frame(last_frame, resized_frame):
            frame_file_path = os.path.join(target_dir, f"resized_frame_{saved_frame_count}.{video_format}.jpg")
            cv2.imwrite(frame_file_path, resized_frame)
            print(f"Extracted and resized frame {saved_frame_count}")
            saved_frame_count += 1
            last_frame = resized_frame  # Update the last saved frame

    cap.release()
    print(f"All unique frames have been extracted, resized, and saved in {target_dir}")


# videos_path = ['pexels_videos_2577 (1080p).mp4', 'pexels_videos_2670 (1080p).mp4', 'pexels_videos_2048206 (1080p).mp4',
#                'video_of_people_walking (Original).mp4','family-football-on-the-beach.mp4','file_example_MOV_640_800kB.mov','file_example_AVI_1920_2_3MG.avi']
videos_path = ['file_example_AVI_1920_2_3MG.avi']
base_path = "/home/romh/PycharmProjects/trail_person/Video_Proccesing/avi"

for video_name in videos_path:
    video_path = os.path.join(base_path, video_name)
    details = get_video_details(video_path=video_path)
    print(details)
    identify_format_and_split_frames(video_path=video_path)
