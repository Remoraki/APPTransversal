import cv2
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import numpy as np


sam_checkpoint = "./sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

image_path = "./textures/texture10.png"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)

# Combine all masks for full segmentation
height, width = image_rgb.shape[:2]
full_segmentation = np.zeros((height, width), dtype=np.uint8)

for mask_data in sam_result:
    mask = mask_data['segmentation']
    full_segmentation = np.maximum(full_segmentation, (mask * 255).astype(np.uint8))

# Save the full segmentation as a PNG file
cv2.imwrite("full_segmentation.png", full_segmentation)

'''mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)'''