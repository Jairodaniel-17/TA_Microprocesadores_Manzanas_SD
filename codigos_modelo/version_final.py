# Importar librerías
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Verificar si GPU está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definir la transformación de datos
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

# Ruta a tus datos
data_path = "manzanitas"

# Crear un conjunto de datos
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Dividir el conjunto de datos en entrenamiento y prueba
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)


# Crear los dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Instanciar el modelo
class CNN(nn.Module):
    def __init__(self):
        """
        Constructor de la clase CNN.
        Inicializa los componentes de la red convolucional.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 256 * 256, 2)  # 2 clases o 2 etiquetas

    def forward(self, x):
        """
        Método forward de la clase CNN.
        Realiza la propagación hacia adelante de la entrada x a través de la red convolucional.

        Args:
            x (torch.Tensor): Tensor de entrada de la red convolucional.

        Returns:
            torch.Tensor: Tensor de salida de la red convolucional.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


# Instanciar el modelo y moverlo a la GPU si está disponible
model = CNN().to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Configurar TensorBoard
writer = SummaryWriter()
# Entrenar el modelo
num_epochs = 10
desired_accuracy = 0.95
# debemos almacenar la precision exacta
precision_final = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Mover datos a la GPU si está disponible
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Escribir pérdida en TensorBoard
        writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)

    # Evaluar el modelo en cada época
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Mover datos a la GPU si está disponible
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    # Escribir precisión en TensorBoard
    writer.add_scalar("Test/Accuracy", accuracy, epoch)

    print(
        f"Época {epoch + 1}/{num_epochs}, Precisión en el conjunto de prueba: {accuracy * 100:.2f}%"
    )
    precision_final = accuracy
    # Verificar si se alcanza la precisión deseada
    if accuracy >= desired_accuracy:
        print(
            f"Precisión minima deseada alcanzada ({desired_accuracy * 100:.2f}%). Deteniendo el entrenamiento. y con una precisión de {accuracy* 100:.2f}%"
        )
        break

# Cerrar el escritor de TensorBoard al final
writer.close()
# Guardar el modelo
# añadir fecha y hora con segundo exacto cuando se esta guardando
# el modelo para que concuerde con lo de tensorboard lo mejor posible
from datetime import datetime

# Obtener la fecha y hora actuales
now = datetime.now()
# Formatear la fecha y hora
formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
# Imprimir o usar la cadena formateada
print(formatted_date_time)
# ademas agregaremos la precisión siendo el modelo + nombre + precision + fecha y hora
nombre = f"modelo_manzanas_{precision_final}_{formatted_date_time}"
torch.save(model.state_dict(), f"{nombre}.pth")
print(f"Nombre del modelo guardo como: {nombre}")

# Exportar el modelo a formato ONNX

import torch

# Define el modelo y carga los pesos entrenados
model = CNN()
nombre_modelo = nombre
model.load_state_dict(torch.load(f"{nombre_modelo}.pth"))
model.eval()

# Define un tensor de ejemplo con las dimensiones correctas
dummy_input = torch.randn(1, 3, 512, 512)

# Exporta el modelo a formato ONNX
torch.onnx.export(model, dummy_input, f"{nombre_modelo}.onnx", verbose=True)
print(f"Modelo guardado exitosamente: {nombre_modelo}.onnx")
