# script using grounding dino to get bounding boxes around concept_only_image

"""STEP 0: import everything"""
import sys
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from fill_sam_testing_terminal.utils.processing import get_obj_name, cxcywh_to_xyxy
import supervision as sv
import numpy as np
import cv2 
import matplotlib.pyplot as plt

print("ദ്ദി(｡•̀ ,<) groundingdino, opencv, and matplotlib modules imported successfully.")


""" Step 1: add all paths """
device = torch.device('cpu')  # Use CPU for inference
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))
dino_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
dino_checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
my_dino_model = load_model(dino_config_path, dino_checkpoint_path).to(device)

print("ദ്ദി(˵•̀ᴗ-˵) Successfully uploaded all paths.")

""" Step 2: run grounding dino on one folder"""
def run_dino_on_folder(sketch_folder_path):
    result_list = [] 
    for sketch_img_path in os.listdir(sketch_only_folder_path):
        # full path to sketch image
        full_path = os.path.join(sketch_folder_path, sketch_img_path)
    
        gd_text_prompt = get_obj_name(sketch_img_path) # returns "text_prompt" for text input of grounding dino
        gd_box_threshold = 0.35
        gd_text_threshold  = 0.25

        image_source, myimage = load_image(full_path)
        detected_boxes, accuracy, obj_name = predict(
            model=my_dino_model,
            image=myimage,
            caption=gd_text_prompt,
            box_threshold=gd_box_threshold,
            text_threshold=gd_text_threshold,
            device="cpu"
        )
        normalized_boxes = detected_boxes.tolist() #cxcywh
        normalized_boxes = cxcywh_to_xyxy(normalized_bboxes) # xyxy
        normalized_bboxes = normalized_bboxes.tolist()
        out_dict = {
                "bboxes": normalized_bboxes,
                "accuracy": accuracy.tolist(),
                "object name": obj_name
                }
    print("(˶ˆᗜˆ˵) Successfully ran grounding dino all paths.")
    return result_list




""" testing 
sketch_folder_path = "test_baseline_data/original_concept_all_lines_centered"










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
sv.plot_image(annotated_image, (10,10))"""