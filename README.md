# TA_Microprocesadores_Manzanas_SD
Librerias que necesitas instalar:
```shell
torch
torchvision
scikit-learn
streamlit
pyserial
```
Proyecto realizado por:

✔️ Mendoza Torres, Jairo Daniel

### Clasificación de Imágenes con Red Neuronal Convolucional (CNN)

Este código implementa una red neuronal convolucional (CNN) en PyTorch para la clasificación de imágenes. El propósito principal es entrenar un modelo capaz de distinguir entre clases específicas de imágenes, en este caso, se menciona la carpeta "manzanitas", lo que sugiere que el modelo puede estar diseñado para clasificar imágenes de manzanas.

#### Contenido del Código:

1. **Importación de Librerías:**
   - Se importan las librerías necesarias, como PyTorch para la construcción y entrenamiento de modelos, y otras utilidades como TensorBoard para la visualización de resultados.

2. **Preparación de Datos:**
   - Se define una serie de transformaciones de datos utilizando `transforms.Compose`, incluyendo el redimensionamiento de imágenes y la conversión a tensores.
   - Se carga el conjunto de datos de imágenes ubicadas en la carpeta "manzanitas" utilizando `datasets.ImageFolder`.
   - El conjunto de datos se divide en conjuntos de entrenamiento y prueba mediante `torch.utils.data.random_split`.

3. **Definición del Modelo (CNN):**
   - Se crea una clase `CNN` que hereda de `nn.Module`, donde se define la arquitectura de la red.
   - La red consta de una capa convolucional, activación ReLU, max pooling, y una capa totalmente conectada para la clasificación binaria.

4. **Configuración del Entrenamiento:**
   - Se instancian el modelo, la función de pérdida (entropía cruzada) y el optimizador (Adam).
   - TensorBoard se configura para la visualización de pérdida y precisión durante el entrenamiento.

5. **Entrenamiento y Evaluación:**
   - Se lleva a cabo el bucle de entrenamiento a lo largo de varias épocas, actualizando los pesos del modelo para minimizar la pérdida.
   - La precisión en el conjunto de prueba se evalúa después de cada época y se registra en TensorBoard.
   - El entrenamiento se detiene si se alcanza la precisión deseada.

6. **Guardado del Modelo:**
   - El modelo entrenado se guarda en un archivo `.pth` que incluye la precisión alcanzada y la fecha y hora del guardado.
   - Esto facilita la identificación de modelos específicos y su rendimiento.

#### Ejecución del Código:

- Asegúrate de tener instaladas las bibliotecas necesarias especificadas en el archivo `requirements.txt`.
- Ejecuta el código para entrenar el modelo y visualizar la pérdida y precisión en TensorBoard.
- El modelo entrenado se guardará en un archivo con un nombre que incluye la precisión y la marca de tiempo.

**Nota:** Modifica las rutas y parámetros según sea necesario para adaptar el código a tus propios datos y requisitos de entrenamiento.
### Conversión del Modelo a Formato ONNX

Después de entrenar y guardar el modelo, se lleva a cabo la conversión del modelo entrenado a formato ONNX para facilitar la inferencia en diversas plataformas y entornos. A continuación, se presenta el fragmento de código adicional:

```python
import torch

# Define el modelo y carga los pesos entrenados
model = CNN()
nombre_modelo = nombre
model.load_state_dict(torch.load(f'{nombre_modelo}.pth'))
model.eval()

# Define un tensor de ejemplo con las dimensiones correctas
dummy_input = torch.randn(1, 3, 512, 512)

# Exporta el modelo a formato ONNX
torch.onnx.export(model, dummy_input, f"{nombre_modelo}.onnx", verbose=True)
print(f"Modelo guardado exitosamente: {nombre_modelo}.onnx")
```

#### Convertir Modelo a Formato ONNX:

1. **Definición y Carga del Modelo:**
   - Se instancia un objeto de la clase `CNN`.
   - Los pesos entrenados del modelo se cargan utilizando `torch.load` a partir del archivo guardado anteriormente.

2. **Definición de Tensor de Ejemplo:**
   - Se crea un tensor de ejemplo (`dummy_input`) con dimensiones coincidentes con el formato de entrada del modelo (en este caso, 3 canales, 512x512 píxeles).

3. **Exportación a Formato ONNX:**
   - Se utiliza `torch.onnx.export` para exportar el modelo a formato ONNX.
   - Se especifica el modelo, el tensor de ejemplo, y el nombre del archivo de salida (`{nombre_modelo}.onnx`).
   - La opción `verbose=True` proporciona información detallada durante la exportación.

4. **Mensaje de Confirmación:**
   - Se imprime un mensaje indicando que el modelo ha sido guardado exitosamente en formato ONNX.

#### Uso del Modelo ONNX:

El archivo ONNX (`{nombre_modelo}.onnx`) ahora puede ser utilizado en entornos compatibles con ONNX para realizar inferencias sin necesidad de tener acceso al código fuente del modelo o a PyTorch. Este formato es especialmente útil para integrar modelos en diferentes frameworks y dispositivos.

# Para ver el tensorboard:
```shell
tensorboard --logdir=runs 
```