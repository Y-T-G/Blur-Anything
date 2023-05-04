from tqdm import tqdm

from utils.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
import numpy as np
import argparse


class TrackingAnything:
    def __init__(self, sam_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.samcontroler = SamControler(
            self.sam_checkpoint, args.sam_model_type, args.device
        )
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True
    ):
        mask, logit, painted_image = self.samcontroler.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument(
        "--port",
        type=int,
        default=6080,
        help="only useful when running gradio applications",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mask_save", default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args
