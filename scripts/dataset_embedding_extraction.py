import torch
import clip
from utils import get_clip_embeddings, get_dino_embeddings
from transformers import AutoProcessor, AutoModel
# This script extracts CLIP and DINO embeddings for images in a dataset directory and saves them.

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Starting CLIP embedding extraction...")
clip_embeddings = get_clip_embeddings(preprocess=preprocess, model=model, device=device, image_dir="data/images")

print("Saving embeddings to files...")
torch.save(clip_embeddings, "embeddings/clip_embeddings.pt")
print("Embedding extraction and saving completed.")

processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)


print("Starting DINO embedding extraction...")
dino_embeddings = get_dino_embeddings(processor=processor, model=model, device=device, image_dir="data/images")

print("Saving embeddings to files...")
torch.save(dino_embeddings, "embeddings/dino_embeddings.pt")
print("Embeddings saved successfully.")