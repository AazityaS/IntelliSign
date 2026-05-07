import sys
sys.path.append("C:/Users/Aaditya/Desktop/Programming/ML project/Sem 4/Restormer")
import torch
import cv2
import os
import numpy as np
from basicsr.models.archs.restormer_arch import Restormer

# ---- LOAD MODEL ----
device = torch.device("cpu")

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4,6,6,8],
    num_refinement_blocks=4,
    heads=[1,2,4,8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias'
)

model.load_state_dict(torch.load("deraining.pth")['params'])
model = model.to(device)
model.eval()


# ---- INPUT / OUTPUT FOLDERS ----
input_folder = "C:/Users/Aaditya/Desktop/Programming/ML project/THOR/THOR/rain/images/val"
output_folder = "derained/images/val"

os.makedirs(output_folder, exist_ok=True)


# ---- PROCESS IMAGES ----
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().permute(1,2,0).cpu().numpy()
    output = (output * 255).clip(0,255).astype(np.uint8)

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(output_folder, img_name), output)

print("Done deraining!")