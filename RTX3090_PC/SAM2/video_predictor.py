"""
Copied script
from https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb
SAM2 example script -- use this to test out the Docker image.
"""

import os
import shutil
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2

sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

parser = argparse.ArgumentParser()
parser.add_argument('--test_scene_dir', required=False, type=str, default="../openCV_data/cup_test")
args = parser.parse_args()
obj_dir = args.test_scene_dir

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Original coloured mask for multiple objects
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def save_binary_mask(mask, binary_img_path):
    # Ensure the mask is 2D; must be sored as 1D image
    h, w = mask.shape[-2:]  # Extract height and width
    mask = mask.reshape(h, w)  # Flatten to 2D if necessary

    # Convert the mask to an 8-bit grayscale image
    mask_grayscale = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_grayscale, mode='L')  # 'L' mode for grayscale
    img.save(binary_img_path)

##########################################################################################################

# `video_dir` a directory of JPEG frames
video_dir = os.path.join(obj_dir, "rgb")
output_dir = os.path.join(obj_dir, "masks")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Two click object selection:
# positive click at (x, y) = (250, 220) & (x, y) = (210, 350)
# sending all clicks (and their labels) to `add_new_points_or_box`

# 01_RealSense_box split0:
# points = np.array([[225, 206], [385, 281], [325, 299]], dtype=np.float32)
# 01_RealSense_box split1:
# points = np.array([[275, 216], [377, 269], [349, 288]], dtype=np.float32)
# 01_RealSense_box split2:
points = np.array([[141, 375], [271, 313], [209, 400]], dtype=np.float32)
# 02_Beaker:
# points = np.array([[322, 199], [302, 266]], dtype=np.float32)
# 03_Mouse:
# points = np.array([[299, 283], [349, 282]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 0], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

'''
# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([610, 390, 800, 500], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)
'''
try:
    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    '''
    show_box(box, plt.gca())
    '''
    # original multiobject multicolour mask
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        # Save binary masks
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            binary_output_path = os.path.join(output_dir, f"{frame_names[out_frame_idx][:-4]}.png")
            save_binary_mask(out_mask, binary_output_path)

    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            output_path = os.path.join(output_dir, frame_names[out_frame_idx])
            # original multiobject multicolour mask
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.show()
finally:
    plt.close()
    '''
    for filename in os.listdir(video_dir):
        # Check if the file ends with .jpg
        if filename.endswith('.jpg'):
            file_path = os.path.join(video_dir, filename)
            # Delete the file
            os.remove(file_path)
    '''