import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn


# Definir la arquitectura de la CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 256 * 256, 2)  # 2 clases

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


# Ruta del modelo guardado
model_path = "apple_classifier_model_512_70.pth"  # Cambia 'ruta_del_modelo.pth' por la ruta correcta

# Instanciar el modelo
model = CNN()

# Cargar el estado del modelo
model.load_state_dict(torch.load(model_path))
model.eval()

# Transformación para la imagen de prueba
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

# Ruta de la imagen de prueba
image_path = "bonita.png"  # Cambia 'ruta_de_la_imagen.jpg' por la ruta correcta

# Cargar la imagen
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Añadir dimensión del lote (batch)

# Realizar la predicción
with torch.no_grad():
    output = model(image)

# Obtener la clase predicha
_, predicted_class = torch.max(output, 1)

# Mostrar el resultado
print(
    f'La imagen es clasificada como: {"Buena" if predicted_class.item() == 0 else "Mala"}'
)
