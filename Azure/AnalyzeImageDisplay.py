import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Azure Vision API credentials
endpoint = "https://visioncomputerproject.cognitiveservices.azure.com/"
key = "7lUp5HZMkNkbdhBqcLXe7ZiM6gwrJDx9WTPv6xPPKfBjgmqXMbxHJQQJ99BCACYeBjFXJ3w3AAAFACOGrlzk"

# Image URL
#image_url = "https://www.mintic.gov.co/portal/715/articles-333040_foto_marquesina.jpg"
#image_url = 'https://i.pinimg.com/736x/3e/ac/9f/3eac9f1f69446f5ccfbba775c63f3254.jpg'
image_url = 'https://i.pinimg.com/736x/92/ce/71/92ce7104580d56a7bd6edda7f6c8de01.jpg'

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Analyze the image
result = client.analyze_from_url(
    image_url=image_url,
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    gender_neutral_caption=True,
)

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image_cv = np.array(image)
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

# Draw the bounding boxes for text
if result.read is not None:
    for line in result.read.blocks[0].lines:
        points = [(int(p.x), int(p.y)) for p in line.bounding_polygon]
        for i in range(len(points)):
            cv2.line(image_cv, points[i], points[(i + 1) % len(points)], (0, 255, 0), 2)
        x, y = points[0]  # Position for text
        cv2.putText(image_cv, line.text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 0), 2)

# Display the caption
if result.caption is not None:
    caption_text = f"Caption: {result.caption.text} (Confidence: {result.caption.confidence:.2f})"
    print(caption_text)
    cv2.putText(image_cv, caption_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Convert back to RGB for Matplotlib display
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Show the image
plt.figure(figsize=(10, 6))
plt.imshow(image_cv)
plt.axis("off")
plt.show()
