import cv2
import numpy as np
import os

# Parameters
root = os.path.join('demo','output','deepprompt_eam3d_all_final_313')
filenames = ["obama_ang_M003_neu_1_001_int0000.mp4", "obama_ang_M003_neu_1_001_int0250.mp4", "obama_ang_M003_neu_1_001_int0500.mp4",
             "obama_ang_M003_neu_1_001_int1000.mp4"]
videos = [os.path.join(root,i) for i in filenames]  # List of video files
frames_per_video = 5  # Number of frames to extract from each video
output_image = "comparison_image.jpg"  # Output image file name
frame_size = (160, 160)  # Resize each frame (width, height)


def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")

    cap.release()
    return frames


def create_comparison_image(videos, frames_per_video, frame_size):
    all_frames = []

    for video in videos:
        frames = extract_frames(video, frames_per_video)
        if len(frames) < frames_per_video:
            print(
                f"Error: Not enough frames extracted from {video}. Skipping...")
            continue
        all_frames.append(frames)

    if len(all_frames) != len(videos):
        print("Error: Not all videos had enough frames. Exiting...")
        return None

    grid_height = len(videos)
    grid_width = frames_per_video
    comparison_image = np.zeros(
        (grid_height * frame_size[1], grid_width * frame_size[0], 3), dtype=np.uint8)

    for i, frames in enumerate(all_frames):
        for j, frame in enumerate(frames):
            y_start = i * frame_size[1]
            x_start = j * frame_size[0]
            comparison_image[y_start:y_start + frame_size[1],
                             x_start:x_start + frame_size[0]] = frame

    return comparison_image


# Generate the comparison image
comparison_image = create_comparison_image(
    videos, frames_per_video, frame_size)
if comparison_image is not None:
    cv2.imwrite(output_image, comparison_image)
    print(f"Comparison image saved as {output_image}")
else:
    print("Failed to create comparison image.")
