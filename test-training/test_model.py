import torch
import cv2
import numpy as np
from model import CSRNet
import matplotlib.pyplot as plt
from ultralytics import YOLO


# load model
model = CSRNet(load_weights=True)
state_dict = torch.load("test-training/weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# load image
img = cv2.imread("test-training/test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize (important!)
img = cv2.resize(img, (640, 480))

# normalize
img = img / 255.0
img = (img - 0.5) / 0.5

# HWC → CHW
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img).unsqueeze(0).float()

# inference
with torch.no_grad():
    output = model(img)
    
#heatmap
# original image (before normalization)
orig = cv2.imread("test-training/test.jpg")
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# resize density map to match image
density_map = output.squeeze().cpu().numpy()
density_map = cv2.resize(density_map, (orig.shape[1], orig.shape[0]))

# normalize heatmap
density_map = density_map / density_map.max()

heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#approx locations


# overlay
overlay = 0.6 * orig + 0.4 * heatmap
overlay = overlay.astype(np.uint8)

plt.imshow(overlay)
plt.title("Crowd Heatmap Overlay")
plt.axis("off")
plt.show()

#dont touch this part, it calculates the count from the density map




count = output.sum().item()

print(f"Predicted Count: {count:.2f}")