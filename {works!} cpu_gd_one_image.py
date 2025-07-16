# script using grounding dino to get bounding boxes around concept_only_image

# == STEP 0: import everything == 
import sys
import os
import torch
# Manually add the GroundingDINO repo to your import path
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))

from groundingdino.util.inference import load_model, load_image, predict, annotate
print("ദ്ദി(｡•̀ ,<) groundingdino modules imported successfully.")
import supervision as sv
import numpy as np
import cv2 
import matplotlib.pyplot as plt
print("٩(ˊᗜˋ*)و OpenCV and Matplotlib imported successfully.")
device = torch.device('cpu')  # Use CPU for inference

# PART ONE: GroundingDINO
# == step 1: add all paths == 
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# checkpoint  is the model weight. we need to download model weight.
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
print("(∩˃o˂∩) Config and checkpoint paths set.")
my_GD_model = load_model(config_path, checkpoint_path).to(device)
print("ദ്ദി(˵•̀ᴗ-˵) Model loaded successfully.")
IMAGE_PATH = "test_baseline_data/original_concept_all_lines_centered/house_Professional5.png"

# == step 2: add all image and text prompts ==
TEXT_PROMPT = "house"
print("૮˶ᵔᵕᵔ˶ა Image path and text prompt set.")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
print("(˶ˆᗜˆ˵)Thresholds set.")

# == step 3: load image and predict ==
image_source, myimage = load_image(IMAGE_PATH)
print("ദ്ദി(｡•̀ ,<) Image loaded successfully.")

detected_boxes, accuracy, obj_name = predict(
    model = my_GD_model,
    image = myimage,
    caption = TEXT_PROMPT,
    box_threshold = BOX_THRESHOLD,
    text_threshold = TEXT_THRESHOLD,
    device = "cpu"
)
print("(˶°ㅁ°)!!", detected_boxes, accuracy, obj_name)

# == step 4: display image with bounding boxes == 
annotated_image = annotate(
    image_source = image_source,
    boxes = detected_boxes,
    logits = accuracy,
    phrases = obj_name
)
print("(˶°ㅁ°)!!", annotated_image.shape)
sv.plot_image(annotated_image, (10,10))

