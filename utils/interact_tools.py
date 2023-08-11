from PIL import Image
import numpy as np
from .base_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter


mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SamControler:
    def __init__(self, sam_pt_checkpoint, sam_onnx_checkpoint, model_type, device):
        """
        initialize sam controler
        """

        self.sam_controler = BaseSegmenter(sam_pt_checkpoint, sam_onnx_checkpoint, model_type, device)
        self.onnx = model_type == "vit_t"

    def first_frame_click(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask=True,
        mask_color=3,
    ):
        """
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        """
        # self.sam_controler.set_image(image)
        neg_flag = labels[-1]

        if self.onnx:
            onnx_coord = np.concatenate([points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = self.sam_controler.predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
            prompts = {
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
            }

        else:
            prompts = {
                "point_coords": points,
                "point_labels": labels,
            }

        if neg_flag == 1:
            # find positive
            masks, scores, logits = self.sam_controler.predict(
                prompts, "point", multimask
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

            prompts["mask_input"] = np.expand_dims(logit[None, :, :], 0)
            masks, scores, logits = self.sam_controler.predict(
                prompts, "both", multimask
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        else:
            # find neg
            masks, scores, logits = self.sam_controler.predict(
                prompts, "point", multimask
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        assert len(points) == len(labels)

        painted_image = mask_painter(
            image,
            mask.astype("uint8"),
            mask_color,
            mask_alpha,
            contour_color,
            contour_width,
        )
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels > 0)], axis=1),
            point_color_ne,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels < 1)], axis=1),
            point_color_ps,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = Image.fromarray(painted_image)

        return mask, logit, painted_image
