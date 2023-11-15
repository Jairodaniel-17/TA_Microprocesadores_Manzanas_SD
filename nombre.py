import torch
import onnx

# Cargar el modelo ONNX existente
modelo_onnx_path = "modelo.onnx"
modelo_onnx = onnx.load(modelo_onnx_path)

# Cambiar el nombre de la salida
nuevo_nombre_salida = "salida_binaria"
for nodo in modelo_onnx.graph.node:
    if "output" in nodo.output:
        nodo.output[0] = nuevo_nombre_salida

# Guardar el modelo ONNX actualizado con el nuevo nombre de salida
modelo_onnx_actualizado_path = "modelo_actualizado.onnx"
onnx.save_model(modelo_onnx, modelo_onnx_actualizado_path)

# Cargar el modelo actualizado en PyTorch si es necesario
modelo_actualizado = onnx.load_model(modelo_onnx_actualizado_path)
