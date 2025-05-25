from mtcnn.mtcnn import MTCNN
import cv2
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np

class EmotionModel:
    def __init__(self, model_path):
        # Load a pre-trained emotion recognition model (e.g., using a pre-trained CNN)
        self.model = torch.load(model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.mtcnn = MTCNN()

        # Define the transform for input images to match the model's expected format
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((48, 48)),  # Ensure image is resized to 48x48
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to match model
        ])

    def imageCrop(image, required_size=(48,48)):
        mtcnn = MTCNN()
        detection = mtcnn.detect_faces(image)[0]
        bonding_box = detection['box']
        resized_bonding_box = [int(bonding_box[0] - bonding_box[2] * 0.1), int(bonding_box[1] - bonding_box[3] * 0.1), int(bonding_box[2] * 1.2), int(bonding_box[3] * 1.2)]
        face = image[resized_bonding_box[0] : resized_bonding_box[0]+resized_bonding_box[2], resized_bonding_box[1] : resized_bonding_box[1]+resized_bonding_box[3]]

        ret = cv2.resize(face, required_size)
        ret = ret.astype('float32')
        mean, std = ret.mean(), ret.std()
        ret = (ret - mean) / std
        return ret
    
    def emoInterpre(self, image):
        cropped_face = self.imageCrop(image)
        if cropped_face is None:
            return "No face detected"
        
        input_tensor = self.transform(cropped_face)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(input_batch)
            _, predicted_class = torch.max(output, 1)
        
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion_labels[predicted_class.item()]
        
        return predicted_emotion