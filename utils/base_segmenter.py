import torch
import numpy as np


class BaseSegmenter:
    def __init__(self, sam_pt_checkpoint, sam_onnx_checkpoint, model_type, device="cuda:0"):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h, vit_t
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert model_type in [
            "vit_b",
            "vit_l",
            "vit_h",
            "vit_t",
        ], "model_type must be vit_b, vit_l, vit_h or vit_t"

        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32

        if (model_type == "vit_t"):
            from mobile_sam import sam_model_registry, SamPredictor
            from onnxruntime import InferenceSession
            self.ort_session = InferenceSession(sam_onnx_checkpoint)
        else:
            from segment_anything import sam_model_registry, SamPredictor

        self.model = sam_model_registry[model_type](checkpoint=sam_pt_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print("repeat embedding, please reset_image.")
            return
        self.predictor.set_image(image)
        self.image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        self.embedded = True
        return

    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert (
            self.embedded
        ), "prediction is called before set_image (feature embedding)."
        assert mode in ["point", "mask", "both"], "mode must be point, mask, or both"

        if mode == "point":
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"],
                multimask_output=multimask,
            )
        elif mode == "mask":
            masks, scores, logits = self.predictor.predict(
                mask_input=prompts["mask_input"], multimask_output=multimask
            )
        elif mode == "both":  # both
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"],
                mask_input=prompts["mask_input"],
                multimask_output=multimask,
            )
        else:
            raise ("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits

    def predict_onnx(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert (
            self.embedded
        ), "prediction is called before set_image (feature embedding)."
        assert mode in ["point", "mask", "both"], "mode must be point, mask, or both"

        if mode == "point":
            ort_inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            masks, scores, logits = self.ort_session.run(None, ort_inputs)

        elif mode == "mask":
            ort_inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
                "mask_input": prompts["mask_input"],
                "has_mask_input": np.ones(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            masks, scores, logits = self.ort_session.run(None, ort_inputs)
        elif mode == "both":  # both
            ort_inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
                "mask_input": prompts["mask_input"],
                "has_mask_input": np.ones(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            masks, scores, logits = self.ort_session.run(None, ort_inputs)
        else:
            raise ("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits
