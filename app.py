import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import streamlit as st
import onnx
import onnxruntime
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


# Cargar un PyTorch model
## Ruta del modelo guardado
# model_path = "apple_classifier_model_512_70.pth"
# model = CNN()
# model.load_state_dict(torch.load(model_path))
# model.eval()

# Cargar modelo ONNX
model_path = "./models/modelo_manzanas_0.9813664596273292_2023-12-14_17-01-13.onnx"
model = onnx.load(model_path)

# Crear una instancia del modelo CNN
cnn_model = CNN()

# Crear una sesión ONNX para realizar inferencia
onnx_session = onnxruntime.InferenceSession(model_path)

# Obtener información sobre las entradas del modelo ONNX
input_info = onnx_session.get_inputs()
# print("Entradas del modelo ONNX:", input_info)

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
        output = onnx_session.run(None, {input_info[0].name: image.numpy()})

    # Obtener la clase predicha
    predicted_class = torch.argmax(torch.tensor(output[0]))

    # Mostrar el resultado
    result = "Buena" if predicted_class.item() == 0 else "Mala"
    if result == "Buena":
        st.success("¡La manzana está buena!")
    else:
        st.error("¡La manzana está mala!")

    # Enviar el resultado al Arduino
    arduino = serial.Serial("COM3", 9600)
    arduino.write(b"b") if result == "Buena" else arduino.write(b"m")
    arduino.close()
