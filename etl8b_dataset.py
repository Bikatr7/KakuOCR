from torch.utils.data import Dataset
from etl8b_parser import read_record_etl8b
from torchvision import transforms

class ETL8BDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = list(read_record_etl8b(file_path))
        self.char_to_index = {char: idx for idx, char in enumerate(set(char for char, _ in self.data))}
        
        # Ensure all images are resized to 64x64
        self.resize = transforms.Resize((64, 64))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        char, img = self.data[idx]
        label = self.char_to_index[char]
        
        # Always resize the image
        img = self.resize(img)
        
        if(self.transform):
            img = self.transform(img)
        
        return img, label

    def get_num_classes(self):
        return len(self.char_to_index)
