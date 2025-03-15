# Importamos las librerías necesarias
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time

# 🔹 Reemplaza con tus credenciales de Azure
subscription_key = "7lUp5HZMkNkbdhBqcLXe7ZiM6gwrJDx9WTPv6xPPKfBjgmqXMbxHJQQJ99BCACYeBjFXJ3w3AAAFACOGrlzk"  # Clave de suscripción de Azure
endpoint = "https://visioncomputerproject.cognitiveservices.azure.com/"      # Endpoint del servicio de Computer Vision

# 🔹 Creamos un cliente de Computer Vision autenticado
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# 🔹 URL de la imagen que queremos analizar (puede ser una imagen en línea)
image_url = "https://www.mintic.gov.co/portal/715/articles-333040_foto_marquesina.jpg"

#Local:
#with open("mi_imagen.jpg", "rb") as image_file:
 #   response = client.read_in_stream(image_file, raw=True)


# 🔹 Llamamos al servicio de OCR para leer el texto de la imagen
print("Extrayendo texto de la imagen...")
response = client.read(url=image_url, raw=True)  # Enviamos la imagen a la API de OCR
operation_location = response.headers["Operation-Location"]  # Obtenemos la ubicación del resultado
operation_id = operation_location.split("/")[-1]  # Extraemos el ID de la operación

# 🔹 Esperamos hasta que la operación se complete
while True:
    result = client.get_read_result(operation_id)  # Consultamos el estado de la operación
    if result.status not in [OperationStatusCodes.running]:  # Si ya no está en ejecución, salimos del bucle
        break
    time.sleep(1)  # Esperamos 1 segundo antes de volver a consultar

# 🔹 Si la operación fue exitosa, mostramos el texto extraído
if result.status == OperationStatusCodes.succeeded:
    for text_result in result.analyze_result.read_results:
        for line in text_result.lines:
            print(line.text)  # Mostramos cada línea de texto extraída