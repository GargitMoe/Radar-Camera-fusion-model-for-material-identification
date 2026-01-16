import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# ===== 1. 加载模型 =====
MODEL_NAME = "prithivMLmods/Minc-Materials-23"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiglipForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def predict_one_image(img_path: str):
    img = load_rgb_image(img_path)
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)              # 走完整分类模型
        logits = outputs.logits                # [1, num_labels]
        probs = F.softmax(logits, dim=-1)[0]   # [num_labels]

    # id2label 映射
    id2label = model.config.id2label

    # top-1 预测
    top1_id = int(torch.argmax(probs).item())
    top1_label = id2label[top1_id]
    top1_prob = float(probs[top1_id].item())

    print(f"Image: {img_path}")
    print(f"Top-1 prediction: {top1_label} (id={top1_id}), prob = {top1_prob:.4f}")

    # 如果你想看 top-5：
    topk = 5
    topk_probs, topk_ids = torch.topk(probs, k=topk)
    print(f"\nTop-{topk} predictions:")
    for rank, (pid, pprob) in enumerate(zip(topk_ids, topk_probs), start=1):
        pid = int(pid.item())
        pprob = float(pprob.item())
        print(f"{rank}. {id2label[pid]} (id={pid})  prob={pprob:.4f}")

    # 如果你真的需要完整 softmax 向量：
    # 返回一个 numpy 数组
    return probs.cpu().numpy()

if __name__ == "__main__":
    # 在这里填你的测试图片路径
    test_img_path = "Scene1/cardboard/1021067_rgb.png"
    probs = predict_one_image(test_img_path)
