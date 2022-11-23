import torch
import clip
from PIL import Image
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
# model = clip.load("ViT-L/14", device=device, whatonly='encode_text')
# pdb.set_trace()

image = preprocess(Image.open("/mnt/petrelfs/huyutao/smallfiles/redright.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["yellow on right", "red on right"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    pdb.set_trace()

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]