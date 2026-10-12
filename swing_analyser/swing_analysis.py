from pathlib import Path
import json
import subprocess
import urllib.request
import cv2
import imageio_ffmpeg
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from metrics import analyse_swing, format_results

# Folders (they sit one level above this script, next to README.md)
ROOT = Path(__file__).parent.parent  # look 2 levels above this script
INPUT_DIR = ROOT / "test_videos"  # go into test_videos folder
OUTPUT_DIR = ROOT / "analysed_swing"  # place the output in analysed_swing
OUTPUT_DIR.mkdir(exist_ok=True)  # make the folder if it doesn't exist

# Which hand holds the racket: "right" or "left"
HAND = "right"

# The pose model file. Downloaded automatically the first time you run this.
MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
if not MODEL_PATH.exists():
    print("Downloading pose model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Which body points to join with lines (numbers are MediaPipe's 33 point IDs)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),   # face
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),                           # shoulders + arms
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),  # hands
    (11, 23), (12, 24), (23, 24),                                               # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                                     # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),                 # feet
]

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
)

# Go through every video in test_videos
for video_path in INPUT_DIR.glob("*"):
    if video_path.suffix.lower() not in [".mp4", ".mov", ".avi"]:
        continue

    print("Processing", video_path.name)
    cap = cv2.VideoCapture(str(video_path))

    # Get video info so the output matches the input
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the output video file (a temporary one first, converted at the end)
    out_path = OUTPUT_DIR / (video_path.stem + "_analysed.mp4")
    results_path = OUTPUT_DIR / (video_path.stem + "_results.json")
    temp_path = OUTPUT_DIR / "temp.mp4"
    out = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # A fresh landmarker for each video (it tracks time, so it can't be reused)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_number = 0
        all_pixels = []  # the body points from every frame, kept for the analysis
        while True:
            success, frame = cap.read()
            if not success:
                break  # end of video

            # MediaPipe wants RGB, OpenCV gives BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Find the body points (timestamp is in milliseconds)
            timestamp = int(frame_number * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp)
            frame_number += 1

            if results.pose_landmarks:
                points = results.pose_landmarks[0]  # first person found
                pixels = [(int(p.x * width), int(p.y * height)) for p in points]
                all_pixels.append(pixels)

                # Draw the lines and points onto the frame
                for a, b in CONNECTIONS:
                    cv2.line(frame, pixels[a], pixels[b], (255, 255, 255), 2)
                for pixel in pixels:
                    cv2.circle(frame, pixel, 4, (0, 0, 255), -1)

            elif all_pixels:
                # Person lost this frame: reuse the last points so the timing stays correct
                all_pixels.append(all_pixels[-1])

            out.write(frame)

    cap.release()
    out.release()

    # Convert to H.264 (QuickTime-friendly) and shrink to 720p so the file is small
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(temp_path), "-c:v", "libx264", "-crf", "28", "-vf", "scale=-2:720", "-pix_fmt", "yuv420p",
                    str(out_path)], check=True)
    temp_path.unlink()
    print("Saved", out_path)

    # Measure the swing and save the results next to the video
    if all_pixels:
        swing_results = analyse_swing(all_pixels, fps, HAND)
        with open(results_path, "w") as file:
            json.dump(swing_results, file, indent=2)
        print(format_results(swing_results))
        print("Saved", results_path)
    else:
        print("No person found in this video, so no results were saved.")