# Blur Anything For Videos

![Mentioned in Awesome OpenVINO](https://camo.githubusercontent.com/9dbe56f475e0abb1d07a9a18d841378c3d40325ae7df300f29491158cf38d013/68747470733a2f2f617765736f6d652e72652f6d656e74696f6e65642d62616467652d666c61742e737667)

Blur Anything is an adaptation of the excellent [Track Anything](https://github.com/gaomingqi/Track-Anything) project which is in turn based on Meta's Segment Anything and XMem. It allows you to blur anything in a video, including faces, license plates, etc.

<div>
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square" href="https://huggingface.co/spaces/Y-T-G/Blur-Anything">
<img src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square">
</a>
</div>

<video src="https://media.githubusercontent.com/media/Y-T-G/Blur-Anything/main/assets/sample-1-blurred-stacked.mp4"></video>

## Get Started
```shell
# Clone the repository:
git clone https://github.com/Y-T-G/Blur-Anything.git
cd Blur-Anything

# Install dependencies: 
pip install -r requirements.txt

# Run the Blur-Anything gradio demo.
python app.py --device [cpu|cuda:0|cuda:1|...] --sam_model_type [vit_t| vit_b|vit_h| vit_l] [--backend [onnx|openvino]]
```

## Features

- FastSAM with ONNX and OpenVINO support.
- Lower memory usage.

## To Do
- [x] Add a gradio demo
- [ ] Add support to use YouTube video URL
- [ ] Add option to completely black out the object
- [ ] Convert XMem to ONNX

## Acknowledgements

The project is an adaptation of [Track Anything](https://github.com/gaomingqi/Track-Anything) which is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem).

Thanks to [PIMS](https://github.com/soft-matter/pims) which is used to process video files while keeping memory usage low.
