#!/usr/bin/env python3
"""Inference: YOLOv8m detection + EfficientNet-B0/CosFace classification for ancient characters."""
import json, os, sys, traceback
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.25"))
IOU_THRESH = float(os.getenv("IOU_THRESH", "0.7"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

# ── Device ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── CosFace (same definition as training) ─────────────
class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        x = nn.functional.normalize(x, dim=1)
        w = nn.functional.normalize(self.weight, dim=1)
        cos_theta = torch.mm(x, w.t())
        # Inference: no margin applied, just scale
        return cos_theta * self.s

# ── Load models (lazy, on first call) ─────────────────
_yolo_model = None
_class_model = None
_label_to_ch = None

def load_models():
    global _yolo_model, _class_model, _label_to_ch

    # 1. YOLO detection model
    yolo_path = MODEL_DIR / "yolo_best.pt"
    print(f"加载 YOLO 模型: {yolo_path}")
    _yolo_model = YOLO(str(yolo_path))
    print("YOLO 模型加载完成")

    # 2. Label mapping
    label_path = MODEL_DIR / "label_to_chinese.json"
    with open(label_path) as f:
        _label_to_ch = json.load(f)
    _label_to_ch = {int(k): v for k, v in _label_to_ch.items()}
    num_classes = max(_label_to_ch.keys()) + 1
    print(f"标签映射加载完成: {num_classes} 个类别")

    # 3. Classification model (EfficientNet-B0 + CosFace)
    model_path = MODEL_DIR / "class_best.pth"
    print(f"加载分类模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    backbone = efficientnet_b0(weights=None)
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone = backbone.to(device).eval()

    feat_dim = ckpt.get("feat_dim", 1280)
    cosface = CosFace(feat_dim, num_classes)
    cosface.load_state_dict(ckpt["arcface_state_dict"])
    cosface = cosface.to(device).eval()

    _class_model = (backbone, cosface)
    print(f"分类模型加载完成 (val_acc={ckpt.get('val_acc', '?'):.4f})")

# ── Image transform (match training) ─────────────────
obc_mean = [0.85233593, 0.85246795, 0.8517555]
obc_std  = [0.31232414, 0.3122127,  0.31273854]

class_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=obc_mean, std=obc_std),
])

@torch.no_grad()
def classify_crop(crop: Image.Image) -> str:
    """Classify a single character crop and return the Chinese character."""
    global _class_model
    backbone, cosface = _class_model
    img = class_transform(crop).unsqueeze(0).to(device)
    features = backbone(img)
    logits = cosface(features)
    pred = logits.argmax(dim=1).item()
    return _label_to_ch.get(pred, f"???")

# ── Find images ──────────────────────────────────────
def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(p for p in fallback.rglob("*") if p.suffix.lower() in suffixes)
    return []

# ── Main ─────────────────────────────────────────────
def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_images()
    print(f"输入目录: {INPUT_DIR}")
    print(f"找到图片: {len(image_paths)}")

    if not image_paths:
        print("未找到图片, 输出空结果")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"已保存: {OUTPUT_FILE}")
        return

    load_models()

    results = {}
    for idx, img_path in enumerate(image_paths, 1):
        if idx == 1 or idx % 50 == 0:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}")

        image_id = img_path.stem
        detections = []
        try:
            # YOLO detection
            yolo_results = _yolo_model(
                str(img_path),
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                device=device.type,
                verbose=False,
            )[0]

            with Image.open(img_path) as full_img:
                orig_w, orig_h = full_img.size

            # Process each detected box
            boxes = yolo_results.boxes
            if boxes is not None and len(boxes) > 0:
                # xyxy format → xywh
                xyxy = boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Clamp to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    if x2 - x1 <= 1 or y2 - y1 <= 1:
                        continue

                    # Crop from original image and classify
                    with Image.open(img_path) as img:
                        crop = img.crop((x1, y1, x2, y2))

                    char = classify_crop(crop)
                    detections.append({
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "text": char,
                    })

            # Sort top-to-bottom, left-to-right
            detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        except Exception as exc:
            print(f"警告: 处理 {img_path.name} 失败: {exc}")
            traceback.print_exc()

        results[image_id] = detections

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUTPUT_FILE}")
    total_chars = sum(len(v) for v in results.values())
    print(f"共处理 {len(results)} 张图片, 识别 {total_chars} 个古文字")

if __name__ == "__main__":
    main()
