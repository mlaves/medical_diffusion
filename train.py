from logging import critical
from tabnanny import check
import torch
from tqdm import tqdm
from fire import Fire

import matplotlib
from matplotlib import pyplot as plt

from dataset import MedMNISTv2, positional_embedding
from models import UNet


#matplotlib.use("Agg")


def test(net, device, alpha_bar, beta, epoch=None):
    timesteps = alpha_bar.shape[0]
    x_t = torch.randn(1, 1, 32, 32).to(device)
    net.eval()
    with torch.inference_mode():
        steps = tqdm(range(0, timesteps)[::-1])
        for t in steps:
            t_embedding = positional_embedding(t, d=32).repeat((1, 1, 32, 1)).to(device)

            # embed_timestep along channel dim
            x_t_embed = torch.concat((x_t, t_embedding), dim=1)

            eps = net(x_t_embed)

            alpha_t = 1-beta[t]
            alpha_bar_t = alpha_bar[t]

            z = torch.randn_like(x_t).to(device)
            sigma_t = torch.sqrt(beta[t]) if t > 0 else 0
            x_t = 1/torch.sqrt(alpha_t)*(x_t-(1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eps) + sigma_t*z
            steps.set_description(f"{t:03d}, {sigma_t:.4f}")

    fig, ax = plt.subplots()
    ax.imshow(x_t.detach().cpu().numpy()[0,0], cmap='gray')
    fig.savefig(f"./logs/valid_{epoch}.png")


def validation(
    net: torch.nn.Module,
    device: torch.device,
    dataset: MedMNISTv2
) -> float:
    from matplotlib import pyplot as plt
    net.eval()
    losses = []
    batches = tqdm(range(len(dataset)))
    criterion = torch.nn.MSELoss()
    with torch.inference_mode():
        for idx, _ in enumerate(batches):
            loss_sample = []
            steps = range(1, dataset._timesteps)[::-1]
            for t in steps:
                x_0, x_t, eps, timestep, alpha_t, alpha_bar_t, t_embedding = dataset._get(idx, t)

                x_t = x_t.unsqueeze(0)
                eps = eps.unsqueeze(0)
                t_embedding  = t_embedding.unsqueeze(0)

                x_t_input = torch.concat((x_t, t_embedding), dim=1).to(device)
                eps_pred = net(x_t_input)

                x_t_1 = 1/torch.sqrt(alpha_t)*(x_t-(1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eps)
                x_t_1_pred = 1/torch.sqrt(alpha_t)*(x_t-(1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eps_pred)

                loss_sample.append(criterion(x_t_1_pred, x_t_1).item())
                print(t, loss_sample[-1])
                if t == 0:
                    print(alpha_t, alpha_bar_t)

            _, ax = plt.subplots(1, 3)
            ax[0].imshow(x_t[0,0].numpy(), vmin=-2, vmax=2)
            ax[1].imshow(x_t_1[0,0].numpy(), vmin=-2, vmax=2)
            ax[2].imshow(x_t_1_pred[0,0].numpy(), vmin=-2, vmax=2)
            plt.show()

            losses.append(torch.sum(torch.tensor(loss_sample)).item())
            batches.set_description(f"loss: {losses[-1]:.4f}")

    return torch.tensor(losses).mean().item()


def main(
    batch_size: int = 1,
    lr: float = 1e-4,
    num_epochs: int = 10,
    timesteps: int = 1000,
    resume: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(in_channels=2, out_channels=1).to(device)
    dataset = MedMNISTv2(split="train_images", timesteps=timesteps)
    dataset_valid = MedMNISTv2(split="val_images", timesteps=timesteps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    if resume:
        checkpoint = torch.load("logs/checkpoint_128.pt", map_location="cpu")
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Resuming from epoch", checkpoint["epoch"])

    losses = []
    for epoch in range(num_epochs):
        net.train()
        loss_train = []
        batches = tqdm(dataloader)
        for _, x_t, eps, _, _, _, t_embedding in batches:
            eps = eps.to(device)

            # embed_timestep along channel dim
            x_t = torch.concat((x_t, t_embedding), dim=1).to(device)

            optimizer.zero_grad()
            eps_pred = net(x_t)
            loss = criterion(eps_pred, eps)
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
            batches.set_description(f"loss: {loss_train[-1]:.4f}")
            break

        losses.append(torch.tensor(loss_train).mean().item())

        validation(net, device, dataset_valid)

        # save model
        torch.save({
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "timesteps": timesteps
        }, f"./logs/checkpoint.pt")

        test(net, device, alpha_bar=dataset.get_alpha_bar(), beta=dataset.get_beta(), epoch=epoch)

if __name__ == "__main__":
    Fire(main)
