# testing sam on one image

# == STEP 0: import everything ==
import os
HOME = os.path.expanduser("~")
os.makedirs(os.path.join(HOME, "weights"), exist_ok=True)
print(":D We did it!  Weights directory created at:", os.path.join(HOME, "weights"))

import torch
import cv2 # for image processing
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

CHECKPOINT_FILENAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = os.path.join(HOME, "weights", CHECKPOINT_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
print(":P Successfully uploaded checkpoint path, device, and model type")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("test_baseline_data/original_concept_all_lines_centered/potato_chip_Professional3.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
print(":D woop woop! Generated mask is successful")