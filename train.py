import argparse
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from model.vit import ViT
from scheduler import CosineAnnealingWarmUpRestarts


def train(model, dataloader, use_amp, scaler, epoch):
    print(f"\nEpoch: {epoch}")
    model.train()
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / (batch_idx + 1)


def test(model, dataloader, scaler, epoch):
    global best_acc
    model.eval()
    test_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Checkpoint
    accuracy = correct / total * 100.
    if accuracy > best_acc:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(
            state, './checkpoint/' + 'ckpt.pt'
        )
        best_acc = accuracy

    # Log
    os.makedirs('log', exist_ok=True)
    content = (time.ctime() + f" Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, val loss: {test_loss:.5f}, acc: {acc:.5f}")
    print(content)
    with open(f'log/log_.txt', 'a') as appender:
        appender.write(content + "\n")

    return test_loss, acc


if __name__ == '__main__':
    # Parsers
    parser = argparse.ArgumentParser(description="Vision Transformer for PyTorch CIFAR10 Training")

    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--n_epochs', default=200, type=int, help="number of epochs")
    parser.add_argument('--patch', default=4, type=int, help="ViT patch")
    parser.add_argument('--dimhead', default=512, type=int)
    parser.add_argument('--amp', default=True, type=bool, help="use automatic mixed precision")

    args = parser.parse_args()

    # Device setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize
    best_acc = 0
    start_epoch = 0

    # Dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    net = ViT()

    # Loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=args.lr)

    # Use custom CosineAnnealingWarmUpRestarts
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.n_epochs)

    # Train
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    list_loss = []
    list_acc = []

    net.cuda()

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss = train(net, train_loader, True, scaler, epoch)
        val_loss, acc = test(net, test_loader, scaler, epoch)

        scheduler.step(epoch - 1)

        list_loss.append(val_loss)
        list_acc.append(acc)
