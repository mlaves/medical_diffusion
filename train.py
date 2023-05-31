import os
from shutil import copy2
from datetime import datetime

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from fire import Fire

import matplotlib

from dataset import VocalFolds
from models import ResUNet, DDPM


matplotlib.use("Agg")


def scale_1_1(x):
    return x*2 - 1


def test(
        ddpm: DDPM,
        sample_size: torch.Size,
    ) -> tuple[torch.Tensor]:
    x_test = ddpm.sample(sample_size)
    img_grid = make_grid(x_test.cpu(), nrow=4, value_range=(-1,1), normalize=True)
    return x_test, img_grid


def validation(
    ddpm: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    for x_0, _ in dataloader:
        x_0 = x_0.to(device)
        break

    return ddpm.validate(x_0)


def main(
    exp_name: str = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    num_epochs: int = 300,
    timesteps: int = 1000,
    beta_schedule: str = "ho",
    img_width: int = 32,
    img_height: int = 32,
    hidden_channels: int = 64,
    stages: int = 2,
    resume: bool = False,
    log_dir: str = "logs",
):
    if not exp_name:
        exp_name = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        print(f"No `exp_name` provided! Using {exp_name}")

    device = torch.device("mps")
    print(f"Using beta_schedule: {beta_schedule}")

    ddpm = DDPM(
        model=ResUNet(
            in_channels=3,
            out_channels=3,
            use_norm_layer=True,
            hidden_channels=hidden_channels,
            stages=stages,
            use_dropout=False
        ),
        device=device,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
    ).to(device)

    # dataset = VocalFolds(split="train_images", timesteps=timesteps)
    # dataset_valid = dataset

    dataset_train = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(size=(img_width, img_height)),
            transforms.ToTensor(),
            transforms.Lambda(scale_1_1)
        ])
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataset_test = CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(size=(img_width, img_height)),
            transforms.ToTensor(),
            transforms.Lambda(scale_1_1)
        ])
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=16,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150)

    exp_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)
    copy2(__file__, exp_dir)
    copy2(os.path.join(os.path.dirname(__file__), "models.py"), exp_dir)

    start_epoch = 0
    losses = []
    valid_losses = []

    if resume:
        checkpoint = torch.load("logs/checkpoint.pt", map_location=device)
        ddpm.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        losses = checkpoint["loss"]
        valid_losses = checkpoint["valid_loss"]
        timesteps = checkpoint["timesteps"]
        print("Resuming from epoch", start_epoch)

    train_pbar = tqdm(range(start_epoch, num_epochs), position=1)
    train_pbar.set_description(f"train: {0.0:.5f}, valid: {0.0:.5f}")
    for epoch in train_pbar:
        ddpm.train()

        loss_train = []
        pbar_batches = tqdm(dataloader_train, position=2, leave=False)
        pbar_batches.set_description(f"train loss: {0.0:.5f}")
        for x_0, _ in pbar_batches:
            # prepare data
            optimizer.zero_grad()
            x_0 = x_0.to(device)

            loss = ddpm.training_step(x_0)
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
            pbar_batches.set_description(f"train loss: {torch.tensor(loss_train).mean():.5f}")

        losses.append(torch.tensor(loss_train).mean().item())
        lr_scheduler.step()

        ddpm.eval()
        if epoch % 5 == 0 or epoch == num_epochs:
            valid_loss = validation(ddpm, device, dataloader_test)
            valid_losses.append(valid_loss)

            x_test, img_grid = test(ddpm, torch.Size((16, 3, img_height, img_width)))

            writer.add_scalar("train/valid_loss", valid_losses[-1], epoch)
            writer.add_image('test/samples', img_grid, epoch)
            writer.add_histogram("test/histogram", x_test.flatten().clip(-5, 5), epoch)

        train_pbar.set_description(f"train: {losses[-1]:.5f}, valid: {valid_losses[-1]:.5f}")
        writer.add_scalar("train/train_loss", losses[-1], epoch)

        # save model
        torch.save({
            "epoch": epoch,
            "net_state_dict": ddpm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses,
            "valid_loss": valid_losses,
            "timesteps": timesteps
        }, os.path.join(exp_dir, "checkpoint.pt"))


if __name__ == "__main__":
    Fire(main)
