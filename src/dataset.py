import random
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.08):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        x = x + noise
        return torch.clamp(x, 0.0, 1.0)

def get_transforms(cfg):
    train_transforms = [
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]

    if cfg.use_blur:
        train_transforms.append(transforms.GaussianBlur(kernel_size=cfg.blur_kernel))

    if cfg.use_noise:
        train_transforms.append(AddGaussianNoise(std=cfg.noise_std))

    train_tf = transforms.Compose(train_transforms)

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
    ])

    return train_tf, eval_tf

def create_dataloaders(cfg):
    set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {cfg.data_dir}")

    train_tf, eval_tf = get_transforms(cfg)

    dataset_train = datasets.ImageFolder(cfg.data_dir, transform=train_tf)
    dataset_eval = datasets.ImageFolder(cfg.data_dir, transform=eval_tf)

    total_size = len(dataset_train)

    max_samples = 2000

    if total_size > max_samples:
        indices = list(range(total_size))
        random.shuffle(indices)
        indices = indices[:max_samples]

        dataset_train = Subset(dataset_train, indices)
        dataset_eval = Subset(dataset_eval, indices)

        total_size = max_samples

    computed_test = int(total_size * cfg.test_split)

    if total_size >= 3000:
        test_size = min(cfg.target_test_size, total_size // 3)
    else:
        test_size = computed_test

    val_size = int(total_size * cfg.val_split)
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        raise ValueError("Too small dataset for a split.")

    generator = torch.Generator().manual_seed(cfg.seed)

    train_idx, val_idx, test_idx = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=generator
    )

    train_dataset = Subset(dataset_train, train_idx.indices)
    val_dataset = Subset(dataset_eval, val_idx.indices)
    test_dataset = Subset(dataset_eval, test_idx.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2
    )

    print("Total used images:", total_size)
    print("Train size:", train_size)
    print("Val size:", val_size)
    print("Test size:", test_size)

    return train_loader, val_loader, test_loader, dataset_train.dataset.classes if isinstance(dataset_train, Subset) else dataset_train.classes