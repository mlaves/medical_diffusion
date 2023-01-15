"""
Implements the diffusion denoising training dataset using MedMNISTv2.
https://medmnist.com
"""

import os
import numpy as np
import torch


class MedMNISTv2(torch.utils.data.Dataset):
    def __init__(
            self,
            split: str,
            timesteps: int = 1000,
            data_root: str = "./data",
    ) -> None:
        super().__init__()
        if split not in ["train_images", "val_images", "test_images"]:
            raise ValueError("Invalid split")
        self._split = split
        self._timesteps = timesteps
        self._data_root = data_root
        self._dataset_name = "chestmnist"

        self.info = {
            "name": "chestmnist",
            "url": "https://zenodo.org/record/6496656/files/chestmnist.npz?download=1",
            "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        }

        self._download()

        self._data = {}
        dataset = np.load(os.path.join(self._data_root, f"{self.info['name']}.npz"))
        for key in dataset.files:
            self._data[key] = torch.from_numpy(dataset[key]).unsqueeze(1)

        # variance schedule
        self._beta = torch.linspace(10e-4, 0.02, self._timesteps)

    def _download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(
                url=self.info["url"],
                root=self._data_root,
                filename=f"{self.info['name']}.npz",
                md5=self.info["MD5"]
            )
        except Exception as ex:
            raise RuntimeError("Error downloading the dataset.") from ex

    def __len__(self):
        return 0

    def __getitem__(self, index):
        t = torch.randint(low=0, high=self._timesteps, size=(1,)).item()
        return self._get(index, t)

    def _get(self, index, t):
        x_0 = self._data[self._split][index].float()
        x_0 = x_0*2/255 - 1  # scale to [-1, 1]

        eps = torch.randn_like(x_0)
        alpha_bar_t = torch.prod(1-self._beta[:t])
        x_t = torch.sqrt(alpha_bar_t)*x_0 + torch.sqrt(1-alpha_bar_t)*eps

        return x_0, x_t, eps, t, alpha_bar_t
