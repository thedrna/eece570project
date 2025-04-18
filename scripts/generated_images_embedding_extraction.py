import torch
import clip
from utils import get_clip_embeddings, get_dino_embeddings
from transformers import AutoProcessor, AutoModel

# This script extracts CLIP and DINO embeddings for images in a generated images directory and saves them.

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)


# Process generated_images/v1-5
print("Processing DINO embeddings for generated_images/v1-5...")
dino_embeddings_v1 = get_dino_embeddings(processor=processor, model=model, device=device, image_dir="generated_images/v1-5")
torch.save(dino_embeddings_v1, "embeddings/dino_embeddings_v1-5.pt")

# Process generated_images/v2-1
print("Processing DINO embeddings for generated_images/v2-1...")
dino_embeddings_v2 = get_dino_embeddings(processor=processor, model=model, device=device, image_dir="generated_images/v2-1")
torch.save(dino_embeddings_v2, "embeddings/dino_embeddings_v2-1.pt")

# Process generated_images/v1-5
print("Processing CLIP embeddings for generated_images/v1-5...")
model, preprocess = clip.load("ViT-B/32", device=device)
clip_embeddings_v1 = get_clip_embeddings(preprocess=preprocess, model=model, device=device, image_dir="generated_images/v1-5")
torch.save(clip_embeddings_v1, "embeddings/clip_embeddings_v1-5.pt")

# Process generated_images/v2-1
print("Processing CLIP embeddings for generated_images/v2-1...")
clip_embeddings_v2 = get_clip_embeddings(preprocess=preprocess, model=model, device=device, image_dir="generated_images/v2-1")
torch.save(clip_embeddings_v2, "embeddings/clip_embeddings_v2-1.pt")
print("All embeddings processed and saved successfully.")
