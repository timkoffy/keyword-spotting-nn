import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KWS12Dataset
from model import KeywordSpottingModel

def training_loop(epochs=10, batch_size=64, lr=0.001, device=torch.device("cpu")):
    train_dataset = KWS12Dataset(subset="training", augment=True)
    val_dataset = KWS12Dataset(subset="validation", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    model = KeywordSpottingModel(num_classes=12).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0 
        train_correct = 0
        train_total = 0

        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss = 0.0 
        val_correct = 0 
        val_total = 0 

        with torch.inference_mode():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                
                outputs = model(mels)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()


        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "kws_model.pth")
    print("Model saved to kws_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    training_loop(epochs=15, batch_size=64, lr=0.003, device=device)
