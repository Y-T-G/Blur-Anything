import os
import cv2
import numpy as np


# resize frames
def resize_frames(frames, size=None):
    """
    size: (w, h)
    """
    if size is not None:
        frames = [cv2.resize(f, size) for f in frames]
        frames = np.stack(frames, 0)

    return frames


# resize frames
def resize_masks(masks, size=None):
    """
    size: (w, h)
    """
    if size is not None:
        masks = [np.expand_dims(cv2.resize(m, size), 2) for m in masks]
        masks = np.stack(masks, 0)

    return masks


# apply gaussian blur to mask with defined strength
def apply_blur(frame, strength):
    blurred = cv2.GaussianBlur(frame, (strength, strength), 0)
    return blurred


# blur frames
def blur_frames_and_write(
    frames, masks, ratio, strength, dilate_radius=15, fps=30, output_path="blurred.mp4"
):
    assert frames.shape[:3] == masks.shape, "different size between frames and masks"
    assert ratio > 0 and ratio <= 1, "ratio must in (0, 1]"

    # --------------------
    # pre-processing
    # --------------------
    masks = masks.copy()
    masks = np.clip(masks, 0, 1)
    kernel = cv2.getStructuringElement(2, (dilate_radius, dilate_radius))
    masks = np.stack([cv2.dilate(mask, kernel) for mask in masks], 0)
    T, H, W = masks.shape
    masks = np.expand_dims(masks, axis=3)  # expand to T, H, W, 1
    # size: (w, h)
    if ratio == 1:
        size = (W, H)
        binary_masks = masks
    else:
        size = [int(W * ratio), int(H * ratio)]
        size = [
            si + 1 if si % 2 > 0 else si for si in size
        ]  # only consider even values
        # shortest side should be larger than 50
        if min(size) < 50:
            ratio = 50.0 / min(H, W)
            size = [int(W * ratio), int(H * ratio)]
        binary_masks = resize_masks(masks, tuple(size))
        frames = resize_frames(frames, tuple(size))  # T, H, W, 3

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for frame, mask in zip(frames, binary_masks):
        blurred_frame = apply_blur(frame, strength)
        masked = cv2.bitwise_or(blurred_frame, blurred_frame, mask=mask)
        processed = np.where(masked == (0, 0, 0), frame, masked)

        writer.write(processed[:, :, ::-1])

    writer.release()

    return output_path
