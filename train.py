import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import logging
from etl8b_dataset import ETL8BDataset
from model import CNN

## Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx % 100 == 0):
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} '
                f'({100. * correct / len(test_loader.dataset):.1f}%)\n')

def main():
    ## Set working directory dynamically to where the script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    file_path = 'ETL8B/ETL8B/ETL8B2C1'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ETL8BDataset(file_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) ## type: ignore
 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=dataset.get_num_classes()).to(device)
    optimizer = optim.Adam(model.parameters()) ## type: ignore

    ## 50 epochs
    for epoch in range(1, 50):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), "etl8b_model.pth")
    torch.save(dataset.char_to_index, "char_to_index.pth")
    logger.info("Model and char_to_index mapping saved")

if(__name__ == "__main__"):
    main()