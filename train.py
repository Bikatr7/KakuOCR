import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
import os
import logging
from etl8b_dataset import ETL8BDataset
from model import CNN
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast

## Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = nn.functional.cross_entropy(output, target, label_smoothing=0.1)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, patience_counter, patience):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info(f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} '
                f'({accuracy:.1f}%)\n'
                f'Patience Counter: {patience_counter} out of {patience}')
    return accuracy

def main():
    ## Set working directory dynamically to where the script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    file_paths = [
        'ETL8B/ETL8B/ETL8B2C1',
        'ETL8B/ETL8B/ETL8B2C2',
        'ETL8B/ETL8B/ETL8B2C3'
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ElasticTransform(alpha=1.0, sigma=0.5)
    ])

    ## Create a unified character set
    all_chars = set()
    for file_path in file_paths:
        dataset = ETL8BDataset(file_path, transform=transform)
        all_chars.update(dataset.char_to_index.keys())
    
    unified_char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
    num_classes = len(unified_char_to_index)
    
    logger.info(f"Total number of unique characters: {num_classes}")

    ## Create datasets with the unified character set
    datasets = [ETL8BDataset(file_path, transform=transform, char_to_index=unified_char_to_index) for file_path in file_paths]
    combined_dataset = ConcatDataset(datasets)
    
    ## Check class distribution
    class_counts = [0] * num_classes
    for _, label in combined_dataset:
        class_counts[label] += 1
    logger.info(f"Class distribution: Min: {min(class_counts)}, Max: {max(class_counts)}, Avg: {sum(class_counts)/len(class_counts):.2f}")
    
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
 
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose='True')

    scaler = GradScaler() ## type: ignore

    best_accuracy = 0
    patience = 20
    patience_counter = 0

    for epoch in range(1, 201):
        train(model, device, train_loader, optimizer, epoch, scaler)
        accuracy = test(model, device, test_loader, patience_counter, patience)
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "etl8b_model_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), "etl8b_model_final.pth")
    torch.save(unified_char_to_index, "char_to_index.pth")
    logger.info("Model and char_to_index mapping saved")

if __name__ == "__main__":
    main()
