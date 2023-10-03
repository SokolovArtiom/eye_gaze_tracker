import albumentations as A
import albumentations.pytorch as AP
import cv2
import h5py
import numpy as np
import torch

INP_SIZE = 64


class EyeGazeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True):
        self.h5data = h5py.File(data_path, "r")
        np.random.seed(42)
        self.idxs = np.random.choice(
            len(self.h5data["image"]),
            size=len(self.h5data["image"]) // 10 * 9,
            replace=False,
        )
        if not is_train:
            self.idxs = list(set(range(len(self.h5data["image"]))) - set(self.idxs))

        np.random.seed()

        if is_train:
            self.augmentations = A.Compose(
                [
                    A.RandomBrightnessContrast(),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    AP.ToTensorV2(),
                ]
            )
        else:
            self.augmentations = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    AP.ToTensorV2(),
                ]
            )

    def __getitem__(self, index):
        img_name = self.h5data["path"][self.idxs[index]]
        label = self.h5data["look_vec"][self.idxs[index]][:2]
        img = self.h5data["image"][img_name][:]

        img = cv2.resize(img, (INP_SIZE, INP_SIZE))

        img -= img.min()
        img = img / img.max() * 255
        img = img.astype(np.uint8)

        img = np.stack([img, img, img], axis=2)

        img = self.augmentations(image=img)["image"]

        label = torch.Tensor(label).float()

        return img, label

    def __len__(self):
        return len(self.idxs)
