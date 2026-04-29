from config import CFG
from dataset import create_dataloaders
from train import train, test

def main():
    cfg = CFG()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(cfg)
    best_path = train(cfg, train_loader, val_loader, class_names)
    test(cfg, test_loader, class_names, best_path)

if __name__ == "__main__":
    main()