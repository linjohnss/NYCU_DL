import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms
import os
import cv2
import numpy as np

def getData(root, mode):
    if mode == 'train':
        df = pd.read_csv(root + '/train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv(root + '/valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "test18":
        df = pd.read_csv(root + '/resnet_18_test.csv')
        path = df['Path'].tolist()
        return path, None

    elif mode == "test50":
        df = pd.read_csv(root + '/resnet_50_test.csv')
        path = df['Path'].tolist()
        return path, None

    elif mode == "test152":
        df = pd.read_csv(root + '/resnet_152_test.csv')
        path = df['Path'].tolist()
        return path, None

# Create a custom PyTorch transform
class CustomTransform:
    def __init__(self, output_size=(112, 112), mode='train'):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, image):
        if self.mode == 'train':
            image = transforms.RandomHorizontalFlip(p=0.5)(image)
            image = transforms.RandomVerticalFlip(p=0.5)(image)
            image = transforms.RandomRotation(45, expand=True)(image)
        
        # Convert to grayscale
        image_gray = transforms.Grayscale()(image)
        # Convert to numpy array for OpenCV operations
        cv_image = np.array(image_gray)
        # Apply threshold to convert to binary image
        _, thresh = cv2.threshold(cv_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area, take the four largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        # Compute minimum and maximum points
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)
        # Crop the image
        image = image.crop((min_x, min_y, max_x, max_y))

        # Resize the image to the output size
        image = transforms.Resize(self.output_size)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        return image

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(root, mode)
        self.mode = mode
        self.train_transform = CustomTransform(mode=mode)
        self.test_transform = CustomTransform(mode=mode)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        if self.img_name[index].startswith("./"):
            self.img_name[index] = self.img_name[index][2:]
        img_path = os.path.join(self.root, self.img_name[index])
        img = Image.open(img_path).convert('RGB')
        
        if self.mode == 'train':
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)

        if self.mode == 'test18' or self.mode == 'test50' or self.mode == 'test152':
            return img
        else:
            return img, self.label[index]