import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from dataset import get_data_loaders
from model_vit import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    
    wandb.init(
        project="human-disease-convit",
        resume="allow",
        config={
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "model": "vit_tiny"
        }
    )

    config = wandb.config

    
    train_loader, val_loader, num_classes = get_data_loaders(
        "data",
        batch_size=config.batch_size
    )

    
    model = get_model(num_classes).to(device)

    
    print("Computing class weights (fast mode)...")
    targets = train_loader.dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )

    
    checkpoint_path = "./checkpoint_vit.pth"
    start_epoch = 0
    best_acc = 0.0

    
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        print(f"Resuming from epoch {start_epoch}")

    
    for epoch in range(start_epoch, config.epochs):

        print(f"\nEpoch {epoch+1}/{config.epochs}")

        model.train()
        total_loss = 0

        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

        
        model.eval()
        correct = 0
        total = 0

        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step(val_acc)

        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./best_model_vit.pth")
            print("Best model saved!")

        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_acc': best_acc,
        }, checkpoint_path)

        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        print(
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()





