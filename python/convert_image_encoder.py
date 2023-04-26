import clip
import torch
import coremltools as ct
import numpy as np
from PIL import Image


def convert():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_encoder = model.visual
    image_encoder.eval()

    # image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)
    image = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(image_encoder, image)
        scale = 1 / (((0.26862954+0.26130258+0.27577711)/3) * 255.0)
        bias = [
            -(0.48145466 / 0.26862954),
            -(0.4578275 / 0.26130258),
            -(0.40821073 / 0.27577711),
        ]

        ct_model = ct.convert(
            traced,
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=image.shape,
                    scale=scale,
                    bias=bias,
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            outputs=[
                ct.TensorType(name="features"),
            ]
        )
        ct_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
        ct_model.save("ClipImageEncoder.mlmodel")


def predict():
    model = ct.models.MLModel("ClipImageEncoder.mlmodel")
    input_image = Image.open("output.jpg")
    result = model.predict({"image": input_image})
    print(result)


convert()
# predict()
