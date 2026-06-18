import model,torch
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Training Accuracy:", accuracy)