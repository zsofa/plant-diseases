import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes, device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            all_labels.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, all_labels, all_preds

def train(cfg, train_loader, val_loader, class_names):
    device = get_device()
    model = build_model(len(class_names), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    run_dir = cfg.runs_dir / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best_model.pt"
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"{cfg.experiment_name} | {epoch}/{cfg.epochs} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    return best_path

def test(cfg, test_loader, class_names, model_path):
    device = get_device()
    model = build_model(len(class_names), device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    run_dir = cfg.runs_dir / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {cfg.experiment_name}")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {cfg.experiment_name}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test loss: {loss:.4f}\n\n")
        f.write(report)

    print(f"{cfg.experiment_name} test accuracy: {acc:.4f}")
    return acc