import cv2
import torch
import numpy as np
import os
from typing import List
from glob import glob

def load_videoFrames(video: str, time_stamp: List[float]):
    """
    Returns the selected frames in a video according to a list of time stamps in seconds.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for ts in time_stamp:
        frame_number = int(fps * ts)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frameAtTime = cap.read()

        if ret:
            frames.append(frameAtTime)
        else:
            print(f"Warning: Could not read frame at {ts} seconds in {video}")

    cap.release()
    return frames

def calcMatrices(frame_annot: str):
    """
    Creates the intrinsic and extrinsic matrices and returns both matrices along with the corresponding timestamp.

    Args:
        frame_annot (str): The line of annotation for a frame.

    Returns:
        time_stamp (float): Time stamp in seconds.
        CameraIntrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        CameraExtrinsics (np.ndarray): 4x4 camera extrinsic matrix with [R|t].
    """
    annotations = frame_annot.strip().split(" ")
    time_stamp = float(annotations[0]) / 1_000_000

    focal_length_x = float(annotations[1])
    focal_length_y = float(annotations[2])
    principal_point_x = float(annotations[3])
    principal_point_y = float(annotations[4])

    CameraIntrinsics = np.array([
        [focal_length_x, 0, principal_point_x],
        [0, focal_length_y, principal_point_y],
        [0, 0, 1]
    ])

    R_t_values = list(map(float, annotations[6:]))
    R_t_matrix = np.array(R_t_values).reshape(3, 4)
    CameraExtrinsics = np.vstack((R_t_matrix, [0, 0, 0, 1]))

    return time_stamp, CameraIntrinsics, CameraExtrinsics

def create2DHV(orb, model, frame: np.ndarray):
    """
    Creates a sparse point cloud by detecting keypoints and estimating their depth.

    Args:
        orb: ORB feature detector.
        model: Monocular depth estimation model.
        frame (np.ndarray): The input image/frame.

    Returns:
        P (np.ndarray): Sparse point cloud of shape (num_points, 3) containing (x, y, d).
    """
    keypoints = orb.detect(frame, None)
    keypoint_coords = np.array([kp.pt for kp in keypoints])

    model.eval()

    input_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
    input_batch = input_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(input_batch)

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=frame.shape[:2][::-1],
        mode='bicubic',
        align_corners=False
    ).squeeze()

    depth_map = depth_map.cpu().numpy()

    x_coords = keypoint_coords[:, 0].astype(int)
    y_coords = keypoint_coords[:, 1].astype(int)

    # Ensure coordinates are within image bounds
    height, width = depth_map.shape
    valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]

    depth_values = depth_map[y_coords, x_coords]

    # Filter out keypoints with invalid depth values
    valid_depth_indices = depth_values > 0
    x_coords = x_coords[valid_depth_indices]
    y_coords = y_coords[valid_depth_indices]
    depth_values = depth_values[valid_depth_indices]

    P = np.vstack((x_coords, y_coords, depth_values)).T

    return P

def process_REALESTATE10K():
    """
    Processes the REALESTATE10K dataset to extract frames, calculate camera matrices,
    generate sparse point clouds, and save them in appropriate directories.
    """
    dataset_dir = 'path/to/REALESTATE10K'
    videos_dir = os.path.join(dataset_dir, 'videos')
    annotations_dir = os.path.join(dataset_dir, 'annotations')

    output_dir = os.path.join(dataset_dir, 'processed_videos')

    os.makedirs(output_dir, exist_ok=True)

    orb = cv2.ORB_create()
    model_type = "MiDaS_large"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    video_files = glob(os.path.join(videos_dir, '*.mp4'))

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"Processing video: {video_name}")

        video_output_dir = os.path.join(output_dir, video_name)
        frames_dir = os.path.join(video_output_dir, 'frames')
        annotations_output_dir = os.path.join(video_output_dir, 'annotations')

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(annotations_output_dir, exist_ok=True)

        annotation_file = os.path.join(annotations_dir, f"{video_name}.txt")

        if not os.path.isfile(annotation_file):
            print(f"No annotation file found for video {video_name}. Skipping.")
            continue

        with open(annotation_file, 'r') as f:
            annotations = f.readlines()

        time_stamps = []
        frame_annotations = []

        for frame_annot in annotations:
            time_stamp, _, _ = calcMatrices(frame_annot)
            time_stamps.append(time_stamp)
            frame_annotations.append(frame_annot)

        frames = load_videoFrames(video_file, time_stamps)

        for idx, (frame, frame_annot) in enumerate(zip(frames, frame_annotations)):
            ts = time_stamps[idx]

            _, CameraIntrinsics, CameraExtrinsics = calcMatrices(frame_annot)

            P = create2DHV(orb, model, frame)

            frame_filename = os.path.join(frames_dir, f'frame_{idx:04d}.png')
            cv2.imwrite(frame_filename, frame)

            intrinsics_filename = os.path.join(annotations_output_dir, f'intrinsics_{idx:04d}.txt')
            np.savetxt(intrinsics_filename, CameraIntrinsics)

            extrinsics_filename = os.path.join(annotations_output_dir, f'extrinsics_{idx:04d}.txt')
            np.savetxt(extrinsics_filename, CameraExtrinsics)

            spc_filename = os.path.join(annotations_output_dir, f'sparse_point_cloud_{idx:04d}.txt')
            np.savetxt(spc_filename, P)

            print(f"Processed frame {idx} for video {video_name}")

    print("Processing complete.")

if __name__ == "__main__":
    process_REALESTATE10K()