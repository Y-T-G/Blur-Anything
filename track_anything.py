import os
from tqdm import tqdm

from utils.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
import numpy as np
import argparse
import cv2

from typing import Optional


class TrackingAnything:
    def __init__(self, sam_pt_checkpoint, sam_onnx_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.sam_pt_checkpoint = sam_pt_checkpoint
        self.sam_onnx_checkpoint = sam_onnx_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.samcontroler = SamControler(
            self.sam_pt_checkpoint, self.sam_onnx_checkpoint,
            args.sam_model_type, args.backend, args.device
        )
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    ):
        mask, logit, painted_image = self.samcontroler.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(
        self,
        images: list,
        template_mask: np.ndarray,
        write: Optional[bool] = False,
        fps: Optional[int] = "30",
        output_path: Optional[str] = "tracking.mp4",
    ):
        masks = []
        logits = []
        painted_images = []

        if write:
            size = images[0].shape[:2][::-1]
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )

        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
            else:
                mask, logit, painted_image = self.xmem.track(images[i])

            masks.append(mask)
            logits.append(logit)

            if write:
                writer.write(painted_image[:,:,::-1])
            else:
                painted_images.append(painted_image)

        if write:
            writer.release()

        return masks, logits, painted_images


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backend",
        type=str,
        default="",
        choices=["onnx", "openvino"],
        help="Specify either `onnx` or `openvino` backend for vit_t model. Not applicable for other models.")
    parser.add_argument("--sam_model_type", type=str, default="vit_t")
    parser.add_argument(
        "--port",
        type=int,
        default=6080,
        help="only useful when running gradio applications",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mask_save", default=False)
    args = parser.parse_args()

    if args.backend in ("onnx", "openvino") and args.sam_model_type != "vit_t":
        print(f" {args.sam_model_type} does not support `onnx` or `openvino` \
              backend. Using PyTorch backend...")

    if args.debug:
        print(args)
    return args
