#!/usr/bin/env python3
"""Warm up YOLO and classification models (cache downloads, run a dummy inference)."""
import json, sys
from pathlib import Path

import torch
from PIL import Image

MODEL_DIR = Path("/app/models")


def warmup():
    print("Warming up models...")

    # 1. YOLO warmup: load model (downloads weights from cache)
    print("[1/2] Loading YOLO model...")
    from ultralytics import YOLO
    yolo = YOLO(str(MODEL_DIR / "yolo_best.pt"))
    print("YOLO model loaded successfully.")

    # 2. Classification model warmup: load model + label mapping
    print("[2/2] Loading classification model...")
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

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
            return cos_theta * self.s

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_DIR / "class_best.pth", map_location=device, weights_only=True)
    backbone = efficientnet_b0(weights=None)
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone = backbone.to(device).eval()

    num_classes = ckpt["num_classes"]
    feat_dim = ckpt.get("feat_dim", 1280)
    cosface = CosFace(feat_dim, num_classes)
    cosface.load_state_dict(ckpt["arcface_state_dict"])
    cosface = cosface.to(device).eval()

    print(f"Classification model loaded (val_acc={ckpt.get('val_acc', '?'):.4f})")

    # Run dummy inference to warm up GPU
    dummy = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        _ = backbone(dummy)
    print("Dummy inference passed — models are ready.")

    print("All models warmed up successfully.")


if __name__ == "__main__":
    warmup()
