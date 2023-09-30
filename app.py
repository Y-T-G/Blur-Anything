import os
import time
import requests
import sys
import json

import gradio as gr
import numpy as np
import torch
import torchvision
import pims

from export_onnx_model import run_export
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

sys.path.append(sys.path[0] + "/tracker")
sys.path.append(sys.path[0] + "/tracker/model")

from track_anything import TrackingAnything
from track_anything import parse_argument

from utils.painter import mask_painter
from utils.blur import blur_frames_and_write


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints...")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Download successful.")

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "False",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [
        ("", ""),
        (
            "Video uploaded. Click the image for adding targets to track and blur.",
            "Normal",
        ),
    ]
    try:
        frames = pims.Video(video_path)
        fps = frames.frame_rate
        image_size = (frames.shape[1], frames.shape[2])

    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": [0] * len(frames),
        "masks": [0] * len(frames),
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps,
    }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(
        video_state["video_name"], video_state["fps"], len(frames), image_size
    )
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return (
        video_state,
        video_info,
        video_state["origin_images"][0],
        gr.update(visible=True, maximum=len(frames), value=1),
        gr.update(visible=True, maximum=len(frames), value=len(frames)),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True, value=operation_log),
    )


def run_example(example):
    return video_input


# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state):
    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(
        video_state["origin_images"][image_selection_slider]
    )

    # update the masks when select a new template frame
    operation_log = [
        ("", ""),
        (
            "Select frame {}. Try click image and add mask for tracking.".format(
                image_selection_slider
            ),
            "Normal",
        ),
    ]

    return (
        video_state["painted_images"][image_selection_slider],
        video_state,
        interactive_state,
        operation_log,
    )


# set the tracking end frame
def set_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [
        ("", ""),
        (
            "Set the tracking finish at frame {}".format(track_pause_number_slider),
            "Normal",
        ),
    ]

    return (
        interactive_state,
        operation_log,
    )


def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state


def get_blur_strength(blur_strength_slider, interactive_state):
    interactive_state["blur_strength"] = blur_strength_slider

    return interactive_state


# use sam to get the mask
def sam_refine(
    video_state, point_prompt, click_state, interactive_state, evt: gr.SelectData
):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(
        video_state["origin_images"][video_state["select_frame_number"]]
    )
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click(
        image=video_state["origin_images"][video_state["select_frame_number"]],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [
        ("", ""),
        (
            "Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment",
            "Normal",
        ),
    ]
    return painted_image, video_state, interactive_state, operation_log


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        mask_dropdown.append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        select_frame, run_status = show_mask(
            video_state, interactive_state, mask_dropdown
        )

        operation_log = [
            ("", ""),
            (
                "Added a mask, use the mask select for target tracking or blurring.",
                "Normal",
            ),
        ]
    except Exception:
        operation_log = [
            ("Please click the left image to generate mask.", "Error"),
            ("", ""),
        ]
    return (
        interactive_state,
        gr.update(
            choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown
        ),
        select_frame,
        [[], []],
        operation_log,
    )


def clear_click(video_state, click_state):
    click_state = [[], []]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [
        ("", ""),
        ("Clear points history and refresh the image.", "Normal"),
    ]
    return template_frame, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]

    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(
            select_frame, mask.astype("uint8"), mask_color=mask_number + 2
        )

    operation_log = [
        ("", ""),
        ("Select {} for tracking or blurring".format(mask_dropdown), "Normal"),
    ]
    return select_frame, operation_log


# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [
        ("", ""),
        (
            "Track the selected masks, and then you can select the masks for blurring.",
            "Normal",
        ),
    ]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][
            video_state["select_frame_number"]: interactive_state["track_end_number"]
        ]
    else:
        following_frames = video_state["origin_images"][
            video_state["select_frame_number"]:
        ]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][
            int(mask_dropdown[0].split("_")[1]) - 1
        ] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(
                template_mask
                + interactive_state["multi_mask"]["masks"][mask_number]
                * (mask_number + 1),
                0,
                mask_number + 1,
            )
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [
            (
                "Error! Please add at least one mask to track by clicking the left image.",
                "Error",
            ),
            ("", ""),
        ]
        # return video_output, video_state, interactive_state, operation_error
    output_path = "./output/track/{}".format(video_state["video_name"])
    fps = video_state["fps"]
    masks, logits, painted_images = model.generator(
        images=following_frames, template_mask=template_mask, write=True, fps=fps,  output_path=output_path
    )
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]:
        video_state["masks"][
            video_state["select_frame_number"]: interactive_state["track_end_number"]
        ] = masks
        video_state["logits"][
            video_state["select_frame_number"]: interactive_state["track_end_number"]
        ] = logits
        video_state["painted_images"][
            video_state["select_frame_number"]: interactive_state["track_end_number"]
        ] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][
            video_state["select_frame_number"]:
        ] = painted_images

    interactive_state["inference_times"] += 1

    print(
        "For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(
            interactive_state["inference_times"],
            interactive_state["positive_click_times"]
            + interactive_state["negative_click_times"],
            interactive_state["positive_click_times"],
            interactive_state["negative_click_times"],
        )
    )

    return output_path, video_state, interactive_state, operation_log


def blur_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("", ""), ("Removed the selected masks.", "Normal")]

    frames = np.asarray(video_state["origin_images"])[
        video_state["select_frame_number"]:interactive_state["track_end_number"]
    ]
    fps = video_state["fps"]
    output_path = "./output/blur/{}".format(video_state["video_name"])
    blur_masks = np.asarray(video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    blur_mask_numbers = [
        int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))
    ]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(blur_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in blur_mask_numbers:
            continue
        blur_masks[blur_masks == i] = 0

    # blur video
    try:
        blur_frames_and_write(
            frames,
            blur_masks,
            ratio=interactive_state["resize_ratio"],
            strength=interactive_state["blur_strength"],
            fps=fps,
            output_path=output_path
        )
    except Exception as e:
        print("Exception ", e)
        operation_log = [
            (
                "Error! You are trying to blur without masks input. Please track the selected mask first, and then press blur. To speed up, please use the resize ratio to scale down the image size.",
                "Error",
            ),
            ("", ""),
        ]

    return output_path, video_state, interactive_state, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.

    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """

    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


# convert to onnx quantized model
def convert_to_onnx(args, checkpoint, quantized=True):
    """
    Convert the model to onnx format.

    Args:
        model (nn.Module): The model to convert.
        output_path (str): The path to save the onnx model.
        input_shape (tuple): The input shape of the model.
        quantized (bool, optional): Whether to quantize the model. Defaults to True.
    """
    onnx_output_path = f"{checkpoint.split('.')[-2]}.onnx"
    quant_output_path = f"{checkpoint.split('.')[-2]}_quant.onnx"

    print("Converting to ONNX quantized model...")

    if not (os.path.exists(onnx_output_path)):
        run_export(
            model_type=args.sam_model_type,
            checkpoint=checkpoint,
            opset=16,
            output=onnx_output_path,
            return_single_mask=True
        )

    if quantized and not (os.path.exists(quant_output_path)):
        quantize_dynamic(
            model_input=onnx_output_path,
            model_output=quant_output_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

    return quant_output_path if quantized else onnx_output_path


# args, defined in track_anything.py
args = parse_argument()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_t": "mobile_sam.pt",
}
SAM_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_t": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)

# initialize SAM, XMem
folder = "checkpoints"
sam_pt_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)

if args.sam_model_type == "vit_t":
    if args.backend not in ("", "onnx", "openvino"):
        print("vit_t only supports `onnx` and `openvino` backends. Falling back to `onnx`")
    sam_onnx_checkpoint = convert_to_onnx(args, sam_pt_checkpoint, quantized=True)
else:
    sam_onnx_checkpoint = ""

model = TrackingAnything(sam_pt_checkpoint, sam_onnx_checkpoint, xmem_checkpoint, args)

title = """<p><h1 align="center">Blur-Anything</h1></p>
    """
description = """<p>Gradio demo for Blur Anything, a flexible and interactive
              tool for video object tracking, segmentation, and blurring. To
              use it, simply upload your video, or click one of the examples to
              load them. Code: <a
              href="https://github.com/Y-T-G/Blur-Anything">https://github.com/Y-T-G/Blur-Anything</a>
              <a
              href="https://huggingface.co/spaces/Y-T-G/Blur-Anything?duplicate=true"><img
              style="display: inline; margin-top: 0em; margin-bottom: 0em"
              src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""


with gr.Blocks() as iface:
    """
    state for
    """
    click_state = gr.State([[], []])
    interactive_state = gr.State(
        {
            "inference_times": 0,
            "negative_click_times": 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {"mask_names": [], "masks": []},
            "track_end_number": None,
            "resize_ratio": 1,
            "blur_strength": 3,
        }
    )

    video_state = gr.State(
        {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "blur_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30,
        }
    )
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        # for user video input
        with gr.Column():
            with gr.Row():
                video_input = gr.Video()
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    resize_info = gr.Textbox(
                        value="You can use the resize ratio slider to scale down the original image to around 360P resolution for faster processing.",
                        label="Tips for running this demo.",
                    )
                    resize_ratio_slider = gr.Slider(
                        minimum=0.02,
                        maximum=1,
                        step=0.02,
                        value=1,
                        label="Resize ratio",
                        visible=True,
                    )

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(
                            value="Get video info", interactive=True, variant="primary"
                        )

                    # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True,
                                visible=False,
                            )
                            remove_mask_button = gr.Button(
                                value="Remove mask", interactive=True, visible=False
                            )
                            clear_button_click = gr.Button(
                                value="Clear Clicks", interactive=True, visible=False
                            )
                            Add_mask_button = gr.Button(
                                value="Add mask", interactive=True, visible=False
                            )
                    template_frame = gr.Image(
                        type="pil",
                        interactive=True,
                        elem_id="template_frame",
                        visible=False,
                    )
                    image_selection_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                        label="Image Selection",
                        visible=False,
                    )
                    track_pause_number_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                        label="Track end frames",
                        visible=False,
                    )

                with gr.Column():
                    run_status = gr.HighlightedText(
                        value=[
                            ("Text", "Error"),
                            ("to be", "Label 2"),
                            ("highlighted", "Label 3"),
                        ],
                        visible=False,
                    )
                    mask_dropdown = gr.Dropdown(
                        multiselect=True,
                        value=[],
                        label="Mask selection",
                        info=".",
                        visible=False,
                    )
                    video_output = gr.Video(visible=False)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(
                            value="Tracking", visible=False
                        )
                        blur_video_predict_button = gr.Button(
                            value="Blur", visible=False
                        )
                    with gr.Row():
                        blur_strength_slider = gr.Slider(
                            minimum=3,
                            maximum=30,
                            step=2,
                            value=3,
                            label="Blur Strength",
                            visible=False,
                        )

    # first step: get the video information
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[video_input, video_state],
        outputs=[
            video_state,
            video_info,
            template_frame,
            image_selection_slider,
            track_pause_number_slider,
            point_prompt,
            clear_button_click,
            Add_mask_button,
            template_frame,
            tracking_video_predict_button,
            video_output,
            mask_dropdown,
            remove_mask_button,
            blur_video_predict_button,
            blur_strength_slider,
            run_status,
        ],
    )

    # second step: select images from slider
    image_selection_slider.release(
        fn=select_template,
        inputs=[image_selection_slider, video_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status],
        api_name="select_image",
    )
    track_pause_number_slider.release(
        fn=set_end_number,
        inputs=[track_pause_number_slider, video_state, interactive_state],
        outputs=[interactive_state, run_status],
        api_name="end_image",
    )
    resize_ratio_slider.release(
        fn=get_resize_ratio,
        inputs=[resize_ratio_slider, interactive_state],
        outputs=[interactive_state],
        api_name="resize_ratio",
    )

    blur_strength_slider.release(
        fn=get_blur_strength,
        inputs=[blur_strength_slider, interactive_state],
        outputs=[interactive_state],
        api_name="blur_strength",
    )

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status],
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[
            interactive_state,
            mask_dropdown,
            template_frame,
            click_state,
            run_status,
        ],
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status],
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status],
    )

    # tracking video from select image and mask
    blur_video_predict_button.click(
        fn=blur_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status],
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status],
    )

    # clear input
    video_input.clear(
        lambda: (
            {
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "blur_masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 30,
            },
            {
                "inference_times": 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": 0,
                "resize_ratio": 1,
                "blur_strength": 3,
            },
            [[], []],
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=[]),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        [],
        [
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button,
            image_selection_slider,
            track_pause_number_slider,
            point_prompt,
            clear_button_click,
            Add_mask_button,
            template_frame,
            tracking_video_predict_button,
            video_output,
            mask_dropdown,
            remove_mask_button,
            blur_video_predict_button,
            blur_strength_slider,
            run_status,
        ],
        queue=False,
        show_progress=False,
    )

    # points clear
    clear_button_click.click(
        fn=clear_click,
        inputs=[
            video_state,
            click_state,
        ],
        outputs=[template_frame, click_state, run_status],
    )
    # set example
    gr.Markdown("##  Examples")
    gr.Examples(
        examples=[
            os.path.join(os.path.dirname(__file__), "./data/", test_sample)
            for test_sample in [
                "sample-1.mp4",
                "sample-2.mp4",
            ]
        ],
        fn=run_example,
        inputs=[video_input],
        outputs=[video_input],
    )
iface.queue(concurrency_count=1)
iface.launch(
    debug=True, enable_queue=True
)
