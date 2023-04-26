"""
Example logic of photo searcher
"""
import os

import clip
import torch
from PIL import Image


FOLDER = "images"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def scan_photos(path):
    photos = []
    images = sorted(os.listdir(path))
    for idx, name in enumerate(images):
        p = os.path.join(path, name)
        img = preprocess(Image.open(p)).unsqueeze(0).to(device)
        photos.append(img)
    return photos


def main():
    photos = scan_photos(FOLDER)
    with torch.no_grad():
        while True:
            keyword = input("Keyword: ")
            text = clip.tokenize(keyword).to(device)
            probs = []
            for idx, each in enumerate(photos):
                prob, _ = model(each, text)
                probs.append({
                    "idx": idx,
                    "prob": prob,
                })

            probs = sorted(probs, key=lambda x: -x['prob'])

            for prob in probs[0:3]:
                print(prob['idx'], prob['prob'])


if __name__ == "__main__":
    main()
