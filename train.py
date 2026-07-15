import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import KWS12Dataset
from model import KeywordSpottingModelV1 as KWS12Model
from log import TrainingStats
import os


def training_loop(epochs=10, batch_size=64, lr=0.001, device=torch.device("cpu")):
    train_dataset = KWS12Dataset(subset="training", augment=True)
    val_dataset = KWS12Dataset(subset="validation", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    model = KWS12Model(num_classes=12).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',       
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

    stats = TrainingStats()
    stats_path = "models/training_stats.pkl"
    if os.path.exists(stats_path):
        stats = TrainingStats.load(stats_path)

    checkpoint_path = "models/checkpoint.pth"
    best_model_path = "models/kws_model.pth"
    
    start_epoch = 0
    best_val_loss = float('inf')

    if stats.history["epochs"]:
        best_val_loss = min(stats.history["val_loss"])
        start_epoch = stats.history["epochs"][-1]

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed training from epoch {start_epoch + 1}")


    for epoch in range(start_epoch, start_epoch + epochs):
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

                val_loss += loss.item() * mels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()


        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        stats.log(epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            best_indicator = " *BEST*"
        else:
            best_indicator = ""

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)

        stats.save(stats_path)

        print(f"Epoch [{epoch+1}/{start_epoch + epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}{best_indicator}")
  
    # plot_stats("training_stats.pkl")


if __name__ == "__main__":
    from config import DEVICE
    print(f"Using device: {DEVICE}")
    training_loop(epochs=30, batch_size=64, lr=0.001, device=DEVICE)
