import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model

# Load trained model
model = load_model("dinov2_classifier.pth")

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    output = model(image)
    predicted_class = output.argmax(dim=1).item()

    class_names = ["dandelion", "grass"]
    print(f"Predicted Class: {class_names[predicted_class]}")

    return "dandelion" if predicted_class == 0 else "grass"

# Example usage
image_path = "/Users/phuongnguyen/Documents/cours_BGD_Telecom_Paris_2024/712_MLOps/dataset_project/test/grass/00000090.jpg"
print(f"Predicted class: {predict(image_path)}")
