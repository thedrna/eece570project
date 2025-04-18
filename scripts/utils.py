import json
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
from tqdm import tqdm


def get_dino_embeddings(processor, model, device, image_dir):
    embeddings = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('png')]
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for fname in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled = last_hidden_state.mean(dim=1).squeeze().cpu()
        embeddings[fname] = pooled
    return embeddings


def get_clip_embeddings(preprocess, model, device, image_dir):
    embeddings = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('png')]
    print(f"Found {len(image_files)} images in '{image_dir}'")
    
    for fname in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, fname)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).squeeze().cpu()
        embeddings[fname] = image_features
    return embeddings


def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.t())


def find_top_matches(gen_embeds, dataset_embeds, top_k=2):
    dataset_fnames = sorted(dataset_embeds.keys())
    dataset_matrix = torch.stack([dataset_embeds[f] for f in dataset_fnames])
    
    scores = cosine_similarity(gen_embeds, dataset_matrix)
    topk = torch.topk(scores, top_k, dim=1)
    top_indices = topk.indices
    top_scores = topk.values

    top_fnames = [[dataset_fnames[i] for i in row] for row in top_indices.tolist()]
    top_scores = top_scores.tolist()

    return list(zip(top_fnames, top_scores))