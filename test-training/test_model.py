import torch
import cv2
import numpy as np
from model import CSRNet
import matplotlib.pyplot as plt
from ultralytics import YOLO


# load model
model = CSRNet(load_weights=True)
state_dict = torch.load("weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# load image
img = cv2.imread("test.jpg")
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
    
    
    
    
#dont touch this part, it calculates the count from the density map

try:
    count = output.sum().item()
except:
    count = -1

print(f"Predicted Count: {count:.2f}")
    
    
    
    
    
#heatmap
# original image (before normalization)
orig = cv2.imread("test.jpg")
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# resize density map to match image
density_map = output.squeeze().cpu().numpy()
density_map = cv2.resize(density_map, (orig.shape[1], orig.shape[0]))

# normalize heatmap
density_map = density_map / density_map.max()

heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# overlay
overlay = 0.6 * orig + 0.4 * heatmap
overlay = overlay.astype(np.uint8)

plt.imshow(overlay)
plt.title("Crowd Heatmap Overlay")
plt.axis("off")
plt.show()




# v2 using yolo approx locations
model = YOLO("yolov8l.pt")

img = cv2.imread("test.jpg")
results = model(img, conf=0.4, iou=0.5)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 3, (0,255,0), -1)



cv2.putText(img, f"Count: {int(count)}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("People Dots", img)
cv2.waitKey(0)






