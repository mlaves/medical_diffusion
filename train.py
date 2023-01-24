import torch
from tqdm import tqdm
from fire import Fire

import matplotlib
from matplotlib import pyplot as plt

from dataset import MedMNISTv2
from models import UNet


matplotlib.use("Agg")


def test(net, device, alpha_bar, beta):
    timesteps = alpha_bar.shape[0]
    x_t = torch.randn(1, 1, 32, 32).to(device)
    with torch.inference_mode():
        steps = tqdm(range(0, timesteps)[::-1])
        for t in steps:
            eps = net(x_t, t/timesteps)

            alpha_t = 1-beta[t]
            alpha_bar_t = alpha_bar[t]

            z = torch.randn(1, 1, 32, 32).to(device)
            sigma_t = torch.sqrt(beta[t]) if t > 0 else 0
            x_t = 1/torch.sqrt(alpha_t)*(x_t-(1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eps) + sigma_t*z
            steps.set_description(f"{t:03d}, {sigma_t:.4f}")

    plt.figure()
    plt.imshow(x_t.detach().cpu().numpy()[0,0], cmap='gray')
    plt.title(f"{str(t)}")
    plt.show()


def main(
    batch_size: int = 1,
    lr: float = 1e-4,
    num_epochs: int = 10,
    timesteps: int = 1000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(in_channels=1, out_channels=1).to(device)
    dataset = MedMNISTv2(split="train_images", timesteps=timesteps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(num_epochs):
        net.train()
        loss_train = []
        batches = tqdm(dataloader)
        for x_0, x_t, eps, t, alpha_t, alpha_bar_t in batches:
            x_t = x_t.to(device)
            eps = eps.to(device)

            optimizer.zero_grad()
            eps_pred = net(x_t, t/timesteps)
            loss = criterion(eps_pred, eps)
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
            batches.set_description(f"loss: {loss_train[-1]:.4f}")

        losses.append(torch.tensor(loss_train).mean().item())

        # save model
        torch.save({
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "timesteps": timesteps
        }, f"checkpoint_{epoch}.pt")

        test(net, device, alpha_bar=dataset.get_alpha_bar(), beta=dataset.get_beta())

if __name__ == "__main__":
    Fire(main)
