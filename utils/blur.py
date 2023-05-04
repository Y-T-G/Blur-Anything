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
def blur_frames(frames, masks, ratio, strength, dilate_radius=15):
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
        size = None
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

    for i, (frame, mask) in enumerate(zip(frames, binary_masks)):
        blurred_frame = apply_blur(frame, strength)
        masked = cv2.bitwise_or(blurred_frame, blurred_frame, mask=mask)
        frames[i] = np.where(masked == (0, 0, 0), frame, masked)

    return frames
