'''Train CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
import wandb
from training import train, test
from utils import select_model, get_lr_scheduler, get_bs_scheduler, get_config_value, get_beta_scheduler
from optim.sgdm import SGDM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    lr = get_config_value(config, "init_lr")
    beta = get_config_value(config, "init_beta")
    epochs = get_config_value(config, "epochs")

    # Dataset Preparation
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Device Setting
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model")
    model = select_model(model_name=model_name, num_classes=100).to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    bs_scheduler, total_steps = get_bs_scheduler(config, trainset_length=len(trainset))
    print(f"total_steps: {total_steps}")

    beta_scheduler, beta_step_type = get_beta_scheduler(config, total_steps)

    optimizer = SGDM(model.parameters(), lr=lr, config=config, total_steps=total_steps)
    lr_scheduler, lr_step_type = get_lr_scheduler(optimizer, config, total_steps, beta_scheduler)
    print(optimizer)

    start_epoch = 0
    steps = 0
    batch_size = bs_scheduler.get_batch_size()

    use_wandb = get_config_value(config, "use_wandb")
    if use_wandb:
        method= "nshb" if get_config_value(config, "nshb") else "shb"
        bs_type = "const_bs" if get_config_value(config, "bs_method")=="constant" else "incr_bs"
        lr_method = get_config_value(config, "lr_method")
        if lr_method in ["constant", "diminishing", "cosine", "linear", "poly"]:
            lr_type = f"decay_lr_{lr_method}"
        elif lr_method == "exp_growth":
            if "lr_max" in config:
                lr_type = f"incr_lr_max{get_config_value(config, 'lr_max')}"
            elif "lr_growth_rate" in config:
                lr_type = f"incr_lr_gamma{get_config_value(config, 'lr_growth_rate')}"
        elif lr_method in ["warmup_const", "warmup_cosine"]:
            if "lr_max" in config:
                lr_type = f"{lr_method}_max{get_config_value(config, 'lr_max')}"
            elif "lr_growth_rate" in config:
                lr_type = f"{lr_method}_gamma{get_config_value(config, 'lr_growth_rate')}"
        else:
            lr_type = "unknown"
        wandb_project_name = "sgdm_CIFAR-100"
        wandb_exp_name = f"{method}_{bs_type}_{lr_type}"
        wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")

    for epoch in range(start_epoch, epochs):
        batch_size = bs_scheduler.get_batch_size()
        beta = beta_scheduler.get_beta()
        print(f'batch size: {batch_size}')
        print(f'learning rate: {lr_scheduler.get_last_lr()[0]}')
        print(f'beta: {beta}')

        train_loss, train_acc, norm_result, last_lr, last_beta = train(model, device, trainset, optimizer, lr_scheduler, lr_step_type, beta_scheduler, beta_step_type, criterion, batch_size)
        test_loss, test_acc = test(model, device, testloader, criterion)
    
        if lr_step_type == "epoch":
            lr_scheduler.step()
        if beta_step_type == "epoch":
            beta_scheduler.step()
        bs_scheduler.step()

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if use_wandb:
            wandb.log({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "norm_result": norm_result,
                    "test_loss": test_loss, "test_acc": test_acc
                    })
        
        if use_wandb:
            wandb.log({"batch_size": batch_size, "learning_rate": last_lr, "beta": last_beta})
