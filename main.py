from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
# 類似typeScript
from typing import Tuple, Optional

class ClassificationModel(nn.Module):
    def __init__(self,num_classes=62):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # 因是灰階圖所以in_channels=1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, num_classes)
        )

    def forward(self, x):
        # 提取特徵
        x = self.feature_extractor(x)
        # 分類
        return self.classifier(x)
    
# 設定數據轉換
def get_transforms() -> transforms.Compose:
    """獲取數據預處理轉換"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(), # 轉換為灰階
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize([0.5],[0.5])
    ])
        
def load_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    transforms = get_transforms()

    train_dataset = datasets.ImageFolder(root='./data/handwriting/augmented_images/augmented_images1', transform=transforms)
    test_dataset = datasets.ImageFolder(root='./data/handwriting/handwritten-english-characters-and-digits/combined_folder/test', transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train(model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module = nn.CrossEntropyLoss(), ) -> list:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.01)

    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # 前向傳播
        out = model(x)
        # 計算損失
        loss = criterion(out, y)
        # 反向傳播
        loss.backward()
        # 更新參數
        optimizer.step()
        # 累積損失
        total_loss += loss.item()
        # 累積正確預測數
        correct += (out.argmax(1) == y).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Average Loss in Training={total_loss / len(train_loader)}, Correct Rate={acc* 100:.2f}")
    return total_loss / len(train_loader), acc

def evaluate(model: nn.Module, 
            test_loader: DataLoader, 
            criterion: nn.Module = nn.CrossEntropyLoss()) -> list:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
    acc = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), acc


def main():
    model = ClassificationModel()
    train_loader, test_loader = load_data()
    for epoch in range(20):
        print(f"Epoch {epoch+1}:")
        train_loss, train_acc = train(model=model, train_loader=train_loader)

    val_loss, val_acc = evaluate(model=model, test_loader=test_loader)
    print(f"final result: "
        f"Average Loss in Training={train_loss:.4f}, "
        f"Average Loss in Testing={val_loss:.4f}, Correct Rate={val_acc* 100:.2f}")

if __name__ == "__main__":
    main()