from torch.utils.data import Dataset
from etl8b_parser import read_record_etl8b
from torchvision import transforms
from PIL import Image

class ETL8BDataset(Dataset):
    def __init__(self, file_path, transform=None, char_to_index=None):
        self.file_path = file_path
        self.transform = transform
        self.data = list(read_record_etl8b(file_path))
        
        if(char_to_index is None):
            self.char_to_index = {char: idx for idx, char in enumerate(set(char for char, _ in self.data))}
        else:
            self.char_to_index = char_to_index
        
        ## Ensure all images are resized to 64x64
        self.resize = transforms.Resize((64, 64))
        
        ## Data augmentation
        self.augment = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomInvert(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        char, img = self.data[idx]
        label = self.char_to_index[char]
        
        ## Ensure img is in 'L' mode (8-bit pixels, black and white)
        if(img.mode != 'L'):
            img = img.convert('L')
        
        ## Always resize the image
        img = self.resize(img)
        
        ## Apply data augmentation
        img = self.augment(img)
        
        if(self.transform):
            img = self.transform(img)
        
        return img, label

    def get_num_classes(self):
        return len(self.char_to_index)
