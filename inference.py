import torch
from torchvision import transforms
from PIL import Image

from vit import VisionTransformer


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model 
model = VisionTransformer()

model.load_state_dict(
    torch.load("VIT/best_vit_model.pth", map_location=device)
)

model.to(device)
model.eval()


class_names = ["ai", "real"]

# Preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs[0][idx].item()

    return class_names[idx], round(confidence * 100, 2)
