from pycocotools.coco import COCO
import os, requests
from tqdm import tqdm
import json
# This script creates a subset of the COCO dataset with images containing 2 to 3 unique categories.

ann_path = "coco_data/annotations/instances_val2017.json" # deleted after getting the subset
img_base_url = "http://images.cocodataset.org/val2017/"
save_dir = "coco_subset" # deleted after getting the subset, moved to data/images
os.makedirs(save_dir, exist_ok=True)

coco = COCO(ann_path)

image_ids = coco.getImgIds()
selected_imgs = []

for img_id in tqdm(image_ids):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    categories = set([ann['category_id'] for ann in anns])
    if 2 <= len(categories) <= 3:
        selected_imgs.append(img_id)
    if len(selected_imgs) == 30: # Limit to 30 images
        break

subset_data = []
for img_id in selected_imgs:
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

    img_url = img_base_url + img_info['file_name']
    img_data = requests.get(img_url).content
    with open(os.path.join(save_dir, img_info['file_name']), 'wb') as f:
        f.write(img_data)

    subset_data.append({
        "file_name": img_info['file_name'],
        "annotations": anns
    })

with open(os.path.join(save_dir, "subset_annotations.json"), 'w') as f:
    json.dump(subset_data, f, indent=2)