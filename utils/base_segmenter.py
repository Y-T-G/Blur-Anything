import torch
import numpy as np


class BaseSegmenter:
    def __init__(self, sam_pt_checkpoint, sam_onnx_checkpoint, model_type,
                 backend, device="cuda:0"):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h, vit_t
        """
        print(f"Initializing BaseSegmenter to {device} with {backend} backend")
        assert model_type in [
            "vit_b",
            "vit_l",
            "vit_h",
            "vit_t",
        ], "model_type must be vit_b, vit_l, vit_h or vit_t"

        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32

        self.backend = backend

        if (model_type == "vit_t"):
            from mobile_sam import sam_model_registry, SamPredictor
            if self.backend == "onnx":
                from onnxruntime import InferenceSession
                self.ort_session = InferenceSession(sam_onnx_checkpoint)
                self.predict = self.predict_onnx_ov
            elif self.backend == "openvino":
                from openvino import Core
                ov_core = Core()
                ov_model = ov_core.read_model(sam_onnx_checkpoint)
                ov_device = "CPU" if device == "cpu" else "AUTO"
                self.ir_model = ov_core.compile_model(model=ov_model,
                                                      device_name=ov_device)
                self.ov_ir = self.ir_model.create_infer_request()
                self.predict = self.predict_onnx_ov
            else:
                raise ("Unsupported Backend")
        else:
            from segment_anything import sam_model_registry, SamPredictor
            self.predict = self.predict_pt

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

    def predict_pt(self, prompts, mode, multimask=True):
        """
        Prediction using PyTorch backend.

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

    def predict_onnx_ov(self, prompts, mode, multimask=True):
        """
        Prediction using ONNX or OpenVINO backend.

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
            inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            if self.backend == "onnx":
                masks, scores, logits = self.ort_session.run(None, inputs)
            elif self.backend == "openvino":
                masks, scores, logits = self.ov_ir.infer(inputs).to_tuple()
            masks = masks > self.predictor.model.mask_threshold

        elif mode == "mask":
            inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": np.zeros((len(prompts["point_labels"]), 2), dtype=np.float32),
                "point_labels": prompts["point_labels"],
                "mask_input": prompts["mask_input"],
                "has_mask_input": np.ones(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            if self.backend == "onnx":
                masks, scores, logits = self.ort_session.run(None, inputs)
            elif self.backend == "openvino":
                masks, scores, logits = self.ov_ir.infer(inputs).to_tuple()
            masks = masks > self.predictor.model.mask_threshold

        elif mode == "both":  # both
            inputs = {
                "image_embeddings": self.image_embedding,
                "point_coords": prompts["point_coords"],
                "point_labels": prompts["point_labels"],
                "mask_input": prompts["mask_input"],
                "has_mask_input": np.ones(1, dtype=np.float32),
                "orig_im_size": prompts["orig_im_size"],
            }
            if self.backend == "onnx":
                masks, scores, logits = self.ort_session.run(None, inputs)
            elif self.backend == "openvino":
                masks, scores, logits = self.ov_ir.infer(inputs).to_tuple()
            masks = masks > self.predictor.model.mask_threshold

        else:
            raise ("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks[0], scores[0], logits[0]