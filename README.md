---
title: Blur Anything
emoji: 💻
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 3.28.1
app_file: app.py
pinned: false
license: mit
---

# Blur Anything For Videos

Blur Anything is an adaptation of the excellent [Track Anything](https://github.com/gaomingqi/Track-Anything) project which is in turn based on Meta's Segment Anything and XMem. It allows you to blur anything in a video, including faces, license plates, etc.

<div>
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square" href="https://huggingface.co/spaces/watchtowerss/Track-Anything">
<img src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square">
</a>
</div>

## Get Started
```shell
# Clone the repository:
git clone https://github.com/Y-T-G/Blut-Anything.git
cd Blur-Anything

# Install dependencies: 
pip install -r requirements.txt

# Run the Blur-Anything gradio demo.
python app.py --device cuda:0
# python app.py --device cuda:0 --sam_model_type vit_b # for lower memory usage
```

## To Do
- [x] Add a gradio demo
- [ ] Add support to use YouTube video URL
- [ ] Add option to completely black out the object

## Acknowledgements

The project is an adaptation of [Track Anything](https://github.com/gaomingqi/Track-Anything) which is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem).
