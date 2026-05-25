import os
import sys
import requests
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd
import numpy as np

# --------------------------------
# CONFIG
# --------------------------------
BASE_URL = "https://huggingface.co/SprintML/tml26_task2/resolve/main"
TARGET_PATH = Path("target_model/weights.safetensors")
TMP_DIR = Path("/tmp/suspect_models")
TMP_DIR.mkdir(exist_ok=True)
NUM_MODELS = 360
OUTPUT_CSV = Path("submission.csv")

# --------------------------------
# MODEL DEFINITION
# --------------------------------
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

# --------------------------------
# LOAD TARGET MODEL
# --------------------------------
print("Loading target model...")
target_sd = load_file(str(TARGET_PATH), device="cpu")
target_vec = torch.cat([v.float().flatten() for v in target_sd.values()])
print(f"Target weight vector size: {target_vec.shape[0]:,}")

# --------------------------------
# SCORING FUNCTION
# Per-layer weight similarity between target and suspect model.
# Combines cosine similarity, exact layer matches, near-exact matches,
# identical parameter ratio, and L2 distance.
# --------------------------------
def compute_scores(target_sd, suspect_sd):
    layer_cosines = []
    layer_l2 = []
    exact_matches = 0
    near_exact = 0
    total_layers = 0
    total_params = 0
    identical_params = 0

    for key in target_sd:
        if key not in suspect_sd:
            continue
        t = target_sd[key].float()
        s = suspect_sd[key].float()
        if t.shape != s.shape:
            continue
        total_layers += 1
        t_flat = t.flatten()
        s_flat = s.flatten()
        total_params += t_flat.shape[0]

        cos = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0)).item()
        layer_cosines.append(cos)

        l2 = torch.norm(t_flat - s_flat).item() / (torch.norm(t_flat).item() + 1e-8)
        layer_l2.append(l2)

        if torch.allclose(t, s, atol=1e-6):
            exact_matches += 1
        if cos > 0.99999:
            near_exact += 1
        identical_params += (torch.abs(t_flat - s_flat) < 1e-6).sum().item()

    if total_layers == 0:
        return 0.5

    mean_cos = np.mean(layer_cosines)
    mean_l2 = np.mean(layer_l2)
    exact_ratio = exact_matches / total_layers
    near_exact_ratio = near_exact / total_layers
    identical_param_ratio = identical_params / (total_params + 1e-8)

    score = (
        0.25 * mean_cos +
        0.25 * exact_ratio +
        0.20 * near_exact_ratio +
        0.15 * identical_param_ratio +
        0.15 * (1.0 - min(mean_l2, 1.0))
    )
    return float(np.clip(score, 0.0, 1.0))

# --------------------------------
# SCORE ALL SUSPECT MODELS
# --------------------------------
scores = []
for i in range(NUM_MODELS):
    fname = f"suspect_{i:03d}.safetensors"
    fpath = TMP_DIR / fname
    url = f"{BASE_URL}/suspect_models/{fname}"
    print(f"[{i+1:03d}/360] Downloading {fname}...", flush=True)
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        fpath.write_bytes(r.content)
    except Exception as e:
        print(f"  Download error: {e}")
        scores.append({"id": i, "score": 0.5})
        continue
    try:
        suspect_sd = load_file(str(fpath), device="cpu")
        score = compute_scores(target_sd, suspect_sd)
        print(f"  score={score:.6f}", flush=True)
    except Exception as e:
        print(f"  Scoring error: {e}")
        score = 0.5
    scores.append({"id": i, "score": score})
    try:
        os.remove(fpath)
    except:
        pass

# --------------------------------
# NORMALIZE AND SAVE
# --------------------------------
df = pd.DataFrame(scores)
min_s = df["score"].min()
max_s = df["score"].max()
df["score"] = (df["score"] - min_s) / (max_s - min_s + 1e-8)
df.to_csv(str(OUTPUT_CSV), index=False)
print(f"Done. Saved {OUTPUT_CSV}")
print(df["score"].describe())