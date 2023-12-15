import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import streamlit as st
import serial


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
model_path = "./models/apple_classifier_model_512_90.pth"
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

st.title("Clasificador de Manzanas")
uploaded_file = st.file_uploader("Sube una imagen de manzana")

if uploaded_file is not None:
    # Mostrar la imagen
    st.image(uploaded_file, caption="Imagen de Manzana", use_column_width=True)

    # Procesar la imagen
    image = Image.open(uploaded_file)
    image = transform(image).unsqueeze(0)  # Añadir dimensión del lote (batch)

    # Realizar la predicción
    with torch.no_grad():
        output = model(image)

    # Obtener la clase predicha
    _, predicted_class = torch.max(output, 1)

    # Mostrar el resultado
    result = "Buena" if predicted_class.item() == 0 else "Mala"
    st.write(f"La imagen es clasificada como: {result}")
    if result == "Buena":
        st.success("¡La manzana está buena!")
    else:
        st.error("¡La manzana está mala!")
    # enviar el resultado al arduino condicional ternario
    arduino = serial.Serial("COM3", 9600)
    arduino.write(b"b") if result == "Buena" else arduino.write(b"m")
    arduino.close()
