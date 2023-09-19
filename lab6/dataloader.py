import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision, torch
import matplotlib.pyplot as plt


class ICLEVRDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        if split == 'train':
            json_path = os.path.join(data_dir, 'train.json')
        elif split == 'test':
            json_path = os.path.join(data_dir, 'test.json')
        elif split == 'new_test':
            json_path = os.path.join(data_dir, 'new_test.json')
        with open(json_path, 'r') as f:
            self.conditions = json.load(f)
        with open(os.path.join(data_dir, 'objects.json'), 'r') as f:
            self.objects = json.load(f)
        if split == 'train':
            self.img_dir = os.path.join(data_dir, 'images')
            self.img_names = list(self.conditions.keys())
            self.labels = list(self.conditions.values())
        else:
            self.labels = list(self.conditions)
        self.transform = self.initialize_transforms()
        self.split = split

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        label = []
        for i in labels:
            label.append(self.objects[i])

        labels = torch.zeros(24)
        for i in label:
            labels[i] = 1.0
            
        if self.split == 'train':
            img_name = self.img_names[idx]
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            image = self.transform(image)
            return image, labels
        
        else:
            return labels

    def initialize_transforms(self):
        """
        Define the image transformations.
        """
        data_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return data_transforms
    
if __name__ == '__main__':
    dataset = ICLEVRDataset(data_dir='dataset/')
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, y = next(iter(train_dataloader))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0])
    plt.savefig('test.png')    