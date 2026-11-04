import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

# ------------------ SETUP ------------------
def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

# ------------------ MAIN ------------------
def main():
    setup()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("Starting DDP AdamW training with 2 GPUs...")

    # ------------------ DATA ------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset_path = os.path.expanduser("~/Project/data/cifar10")

    train_set = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_set)
    test_sampler  = DistributedSampler(test_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=64, num_workers=4, pin_memory=True, sampler=train_sampler)
    test_loader  = DataLoader(test_set, batch_size=64, num_workers=4, pin_memory=True, sampler=test_sampler)

    # ------------------ MODEL ------------------
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    num_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ------------------ TRAIN ------------------
    total_start = time.time()

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        start_epoch = time.time()
        correct, total = 0, 0

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}: Train Acc = {100 * correct / total:.2f}%, Time = {time.time()-start_epoch:.2f}s")

    if rank == 0:
        print(f"DDP Training Completed. Total time: {time.time()-total_start:.2f}s")

    # ------------------ TEST (rank 0 only) ------------------
    if rank == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
     # ---------- Save Model (Rank 0 Only) ----------
        save_path = "/home/du215227/Project/model/resnet18_ddp_adamw.pth"
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved to: {save_path}")


    cleanup()

if __name__ == "__main__":
    main()

