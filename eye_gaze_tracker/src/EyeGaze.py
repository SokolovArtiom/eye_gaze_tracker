import cv2
import timm
import torch
import numpy as np

import albumentations as A
import albumentations.pytorch as AP

INP_SIZE = 64


class GazeModel():
    def __init__(self, model_name, weights_path, device):
        super(GazeModel, self).__init__()

        self.device = device

        if model_name == "mobile_net":
            self.model = timm.create_model(
                'mobilenetv3_small_100', pretrained=False, num_classes=2)
        else:
            raise "model is not supported"

        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(device)
        self.model.eval()

        self.process_image = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            AP.ToTensorV2()
        ])

    def predict(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (INP_SIZE, INP_SIZE))

        image -= image.min()
        image = image / image.max() * 255
        image = image.astype(np.uint8)

        image = np.stack([image, image, image], axis=2)
        image = self.process_image(image=image)["image"].view(
            1, 3, INP_SIZE, INP_SIZE)
        image = image.to(self.device)

        predictions = torch.tanh(self.model(image))

        return predictions.view(1, 2).detach().cpu().numpy()[0]
