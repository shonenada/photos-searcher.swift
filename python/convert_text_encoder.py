import clip
import torch
import coremltools as ct


class TextEncoderModel(torch.nn.Module):
    """
    Text Encode from CLIP
    """

    def __init__(self, clipmodel):
        super(TextEncoderModel, self).__init__()
        self.transformer = clipmodel.transformer
        self.token_embedding = clipmodel.token_embedding
        self.positional_embedding = clipmodel.positional_embedding
        self.text_projection = clipmodel.text_projection
        self.ln_final = clipmodel.ln_final

    def forward(self, text):
        dtype = torch.float32
        x = self.token_embedding(text).type(dtype)
        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x .permute(1, 0, 2)
        x = self.ln_final(x).type(dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x / x.norm(dim=1, keepdim=True)
        return x.t()


def convert():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model, _ = clip.load("./clip_finetune_best_model.pt", device=device)
    model = TextEncoderModel(clipmodel=clip_model)
    model.eval()

    text = clip.tokenize("random string")
    traced = torch.jit.trace(model, text)

    ct_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="text", shape=text.shape)
        ],
        outputs=[
            ct.TensorType(name="features"),
        ]
    )
    ct_model.save("ClipTextEncoder.mlmodel")


def predict():
    model = ct.models.MLModel("ClipTextEncoder.mlmodel")
    text = clip.tokenize(["a dog"])
    result = model.predict({"text": text})
    print(result)


def loop_predict():
    model = ct.models.MLModel("ClipTextEncoder.mlmodel")
    while True:
        raw = input("> ")
        text = clip.tokenize([raw])
        result = model.predict({"text": text})
        print(result)


convert()
# predict()
# loop_predict()
