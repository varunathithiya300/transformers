import torch
from PIL import Image
import torchvision.transforms as transforms
from zero_dce import ZeroDCE  # You need to download the Zero-DCE model

# Load model
model = ZeroDCE()
model.load_state_dict(torch.load('zero_dce.pth', map_location=torch.device('cpu')))
model.eval()

# Load image
image = Image.open('D:\knowledge_sharing\me with again.jpg')
transform = transforms.Compose([transforms.ToTensor()])
input_image = transform(image).unsqueeze(0)

# Enhance image
with torch.no_grad():
    output = model(input_image)
output_image = transforms.ToPILImage()(output.squeeze())

# Save enhanced image
output_image.save('brightened_image.jpg')
print("Low-light enhancement complete!")
