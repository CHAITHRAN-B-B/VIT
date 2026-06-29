import os
import warnings
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from vit import build_model_for_checkpoint

MODEL_PATH = "VIT/models/best_vit_v10.pth"
CHECKPOINT_CONFIGS: dict = {
    # "best_vit_v8.pth":       {"threshold": 0.55},
    # "best_vit_combined.pth": {"threshold": 0.50},
}

DEFAULT_REAL_THRESHOLD = 0.60   # predict 'real' only if P(real) >= this value


# Family B & C (torchvision vit_b_16): matches vitv8/v10 val_test_transforms
_TRANSFORM_TORCHVISION = transforms.Compose([
    transforms.Resize(256),         # resize shorter side → 256
    transforms.CenterCrop(224),     # crop centre 224×224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Family A (custom VisionTransformer): direct 224×224 resize
_TRANSFORM_CUSTOM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _is_custom_family(checkpoint_name: str) -> bool:
    """Return True for checkpoints that use the hand-rolled VisionTransformer."""
    n = checkpoint_name.lower()
    return "combined" in n or ("model" in n and "v" not in n)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
class_names  = ["ai", "real"]

# Module-level state (mutated by load_model)
model              = None
transform          = None
REAL_CONFIDENCE_THRESHOLD = DEFAULT_REAL_THRESHOLD
_active_checkpoint = None


def load_model(checkpoint_path: str = MODEL_PATH) -> None:

    global model, transform, REAL_CONFIDENCE_THRESHOLD, _active_checkpoint

    abs_path = checkpoint_path if os.path.isabs(checkpoint_path) \
               else os.path.join(_SCRIPT_DIR, checkpoint_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {abs_path}\n"
            f"Available checkpoints in VIT/models/:\n" +
            "\n".join(
                f"  {f}" for f in sorted(os.listdir(
                    os.path.join(_SCRIPT_DIR, "VIT", "models")
                )) if f.endswith(".pth")
            )
        )

    basename = os.path.basename(abs_path)

    # ── 1. Build the right architecture ───────────────────────────────────────
    new_model = build_model_for_checkpoint(abs_path, num_classes=2)
    new_model.load_state_dict(
        torch.load(abs_path, map_location=device, weights_only=True)
    )
    new_model.to(device)
    new_model.eval()

    # ── 2. Pick matching transform ─────────────────────────────────────────────
    new_transform = _TRANSFORM_CUSTOM if _is_custom_family(basename) \
                    else _TRANSFORM_TORCHVISION

    # ── 3. Apply per-checkpoint config overrides ───────────────────────────────
    cfg = CHECKPOINT_CONFIGS.get(basename, {})
    new_threshold = cfg.get("threshold", DEFAULT_REAL_THRESHOLD)

    # Commit atomically
    model                     = new_model
    transform                 = new_transform
    REAL_CONFIDENCE_THRESHOLD = new_threshold
    _active_checkpoint        = abs_path

    print(
        f"[inference] Loaded  : {basename}\n"
        f"[inference] Family  : {'A – custom VisionTransformer' if _is_custom_family(basename) else 'B/C – torchvision vit_b_16'}\n"
        f"[inference] Device  : {device}\n"
        f"[inference] Threshold: P(real) >= {new_threshold:.0%}"
    )

load_model(MODEL_PATH)

def tensor_to_base64_cmap(tensor, cmap="gray"):
    import numpy as np
    from PIL import Image as PILImage

    colormap = plt.get_cmap(cmap)
    rgba = colormap(tensor.numpy())
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

    buffered = BytesIO()
    PILImage.fromarray(rgb).save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def predict_image(image: Image.Image):

    image      = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    with torch.no_grad():
        # Test-Time Augmentation: average original + horizontal flip
        logits_orig = model(img_tensor)
        logits_flip = model(torch.flip(img_tensor, dims=[3]))
        logits_avg  = (logits_orig + logits_flip) / 2.0
        probs       = F.softmax(logits_avg, dim=1)[0]       # [2]

    prob_real = probs[1].item()
    prob_ai   = probs[0].item()

    if prob_real >= REAL_CONFIDENCE_THRESHOLD:
        label      = "real"
        confidence = round(prob_real * 100, 2)
    else:
        label      = "ai"
        confidence = round(prob_ai * 100, 2)

    return label, confidence
