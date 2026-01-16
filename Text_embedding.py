import torch
from transformers import AutoTokenizer, SiglipModel
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/siglip-base-patch16-224"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
siglip = SiglipModel.from_pretrained(MODEL_NAME).to(DEVICE)
siglip.eval()

texts = [
    "a piece of shiny metal surface",
    "a sheet of transparent glass",
    "a plastic object",
    "a cardboard box"
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(DEVICE)

with torch.no_grad():
    text_emb = siglip.get_text_features(**inputs)   # 只跑文本塔
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    print(text_emb.shape)

