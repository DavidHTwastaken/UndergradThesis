import cv2
import numpy as np
import os
import glob
# Parameters
root = os.path.join('demo','output','deepprompt_eam3d_all_final_313')
# filenames = ["obama_ang_M003_neu_1_001_int0000.mp4", "obama_ang_M003_neu_1_001_int0250.mp4", "obama_ang_M003_neu_1_001_int0500.mp4",
#              "obama_ang_M003_neu_1_001_int1000.mp4"]
# videos = [os.path.join(root,i) for i in filenames]  # List of video files
videos = glob.glob('*.mp4') 
frames_per_video = 5  # Number of frames to extract from each video
output_image = "comparison_image.jpg"  # Output image file name
frame_size = (160, 160)  # Resize each frame (width, height)
labels = ['Original', 'EAT', 'Audio2Head', 'Combined', 'Combined (Neutral)']

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames in {video_path}: {total_frames}")
    frame_indices = np.linspace(0, total_frames - 2, num_frames, dtype=int)
    # print(f"Extracting frames at indices: {frame_indices}")
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


# def add_row_labels(image, labels, frame_size, font_scale=0.6, thickness=1, margin=10):
#     """Add labels to the left side of each row of the image grid."""
#     labeled_width = image.shape[1] + 120  # space for labels
#     labeled_img = np.ones(
#         (image.shape[0], labeled_width, 3), dtype=np.uint8) * 255
#     labeled_img[:, 120:] = image  # paste original image to the right

#     for i, label in enumerate(labels):
#         y = i * frame_size[1] + frame_size[1] // 2 + 5  # vertically centered
#         cv2.putText(labeled_img, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

#     return labeled_img


def add_row_labels(image, labels, frame_size, font_scale=0.6, thickness=1, margin=10, label_width=120):
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled_width = image.shape[1] + label_width
    labeled_img = np.ones(
        (image.shape[0], labeled_width, 3), dtype=np.uint8) * 255
    labeled_img[:, label_width:] = image  # paste original image to the right

    for i, label in enumerate(labels):
        y_top = i * frame_size[1]
        wrapped = wrap_text(label, max_width=label_width - 2 * margin,
                            font=font, font_scale=font_scale, thickness=thickness)
        for j, line in enumerate(wrapped):
            y = y_top + margin + j * int(20 * font_scale)
            cv2.putText(labeled_img, line, (margin, y),
                        font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    return labeled_img


def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines
# Generate the comparison image
comp_img = create_comparison_image(
    videos, frames_per_video, frame_size)
# comp_img = add_row_labels(comp_img, labels, frame_size)
if comp_img is not None:
    cv2.imwrite(output_image, comp_img)
    print(f"Comparison image saved as {output_image}")
else:
    print("Failed to create comparison image.")

