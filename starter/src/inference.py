import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def model_fn(model_dir):
    """
    Load the PyTorch model from the model_dir.
    """
    model_path = f"{model_dir}/model.pt"
    model = torch.load(model_path)  # Load the entire serialized model
    model.eval()  # Set the model to evaluation mode
    return model



def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')

    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint

    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body)).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return preprocess(image).unsqueeze(0)
    elif content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        data = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {data}')
        image = Image.open(io.BytesIO(data['image'].encode('utf-8'))).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    

# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction.argmax(dim=1).item()

def output_fn(prediction, accept="application/json"):
    return json.dumps({"prediction": prediction})