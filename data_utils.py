from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import math


FPS_MAP = {
    "caucafall": 20,
    "cmdfall": 20,
    "edf": 30,
    "gmdcsa24": 30,
    "le2i": 25,
    "mcfd": 30,
    "occu": 30,
    "up_fall": 18,
    "OOPS": 30,
}

def get_fps(path):
    for k, v in FPS_MAP.items():
        if k in path:
            return v
    raise ValueError("FPS not found for {}".format(path))


excluded_samples = [
    # train
    "mcfd/chute12/cam3",
    "cmdfall/colors/S35P11K1",
    "cmdfall/colors/S34P27K7",
    # test
    "cmdfall/colors/S31P44K6",
    "cmdfall/colors/S49P4K7",
    "cmdfall/colors/S27P4K7",
]


def subsample(original_vector: np.ndarray, original_fps: float, target_fps: float) -> np.ndarray:

    num_original_frames = original_vector.shape[0]
    assert num_original_frames > 0, "Original vector must have at least one frame."

    num_target_frames = int(np.floor((num_original_frames - 1) * target_fps / original_fps)) + 1


    # Indices for the new target vector (0, 1, ..., num_target_frames-1)
    target_frame_indices = np.arange(num_target_frames)

    # Calculate corresponding indices to pick from the original vector
    # i_orig = round(i_target * original_fps / target_fps)
    # Note: np.round rounds to the nearest even number for .5 cases (e.g., np.round(2.5) == 2).
    # If different rounding is needed (e.g., round half up), use np.floor(x + 0.5).
    original_indices_to_select = np.round(target_frame_indices * original_fps / target_fps)
    original_indices_to_select = original_indices_to_select.astype(int)

    # Ensure indices are within the bounds of the original vector
    original_indices_to_select = np.clip(original_indices_to_select, 0, num_original_frames - 1)

    subsampled_vector = original_vector[original_indices_to_select]

    return subsampled_vector


def load_feature(feature_path="S10P12K1.h5"):
    with h5py.File(feature_path, "r") as f:
        # Get feature count without loading all features
        feature = f["features"][...]

    return feature


def load_temporal_labels(annotations_file="full.csv"):
    """Load temporal segmentation labels from CSV and create segment index."""

    df = pd.read_csv(annotations_file)
    samples = defaultdict(list)

    for _, row in df.iterrows():
        path = row.iloc[0]  # something like cvhci_fall/caucafall/video/adl/HopS6.mp4
        path = path.replace("cvhci_fall/", "").replace("video/", "").replace(".mp4", "")
        # Fix specific case sensitivity issue: GMDCSA24 -> gmdcsa24
        path = path.replace("GMDCSA24", "gmdcsa24")
        label = row.iloc[1]
        start = row.iloc[2]
        end = row.iloc[3]

        # Add to path-based dictionary
        segment = (float(start), float(end), int(label))
        samples[path].append(segment)
    return samples


def convert_segments_to_segmentations(temporal_labels, num_frames, fps, default_class_id=-1):
    sorted_labels = sorted(temporal_labels, key=lambda x: x[0])
    frame_vector = np.full(num_frames, default_class_id, dtype=np.int32)

    # Fill the vector based on sorted labels
    for start_sec, end_sec, class_id in sorted_labels:
        if end_sec <= start_sec:  # Skip zero or negative duration segments after warning
            continue

        start_frame = math.floor(start_sec * fps)
        end_frame = math.ceil(end_sec * fps)

        # Clamp indices to the vector bounds
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)  # Slice is exclusive at the end

        frame_vector[start_frame:end_frame] = class_id

    return frame_vector


if __name__ == "__main__":
    # feature = load_feature()
    # load_temporal_labels()
    df = pd.read_csv("results/val_cs.csv")
    first_column_name = df.columns[0]
    data_list = df[first_column_name].tolist()
    pass