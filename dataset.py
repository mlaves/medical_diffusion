import os
import subprocess
from glob import glob
from shutil import copy2
from tempfile import TemporaryDirectory
from functools import partial
import numpy as np
import torch
from skimage import io


def _preprocess(img, size: tuple):
    img = img.float()*2/255 - 1  # scale to [-1, 1]
    img = torch.nn.functional.interpolate(img[None,], size=size, mode="bilinear")[0]
    return img


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
        self._dataset_name = "chestmnist"
        self._data_root = os.path.join(data_root, self._dataset_name)

        self.info = {
            "name": "chestmnist",
            "url": "https://zenodo.org/record/6496656/files/chestmnist.npz",
            "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        }

        self._download()

        self._data = {}
        dataset = np.load(os.path.join(self._data_root, f"{self.info['name']}.npz"))
        for key in dataset.files:
            self._data[key] = torch.from_numpy(dataset[key]).unsqueeze(1)

        # variance schedule
        self._alpha_bar = compute_alpha_bar(self._timesteps)
        self._beta = 1-(self._alpha_bar[1:]/self._alpha_bar[:-1])
        self._beta = torch.cat((self._beta[0:1], self._beta)).clip(0, 0.999)

        # preprocess data
        _preprocess_vmapped = torch.func.vmap(
            partial(_preprocess, size=(32, 32)),
            in_dims=0,
            out_dims=0
        )
        self._data[self._split] = _preprocess_vmapped(self._data[self._split])

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

    def get_beta(self):
        return self._beta.clone()

    def get_alpha_bar(self):
        return self._alpha_bar.clone()

    def __len__(self):
        return self._data[self._split].shape[0]

    def __getitem__(self, index):
        t = torch.randint(low=0, high=self._timesteps, size=(1,))
        return self._get(index, t)

    def _get(self, index, timestep):
        x_0 = self._data[self._split][index]
        eps = torch.randn_like(x_0)

        alpha_bar_t = self._alpha_bar[timestep]
        alpha_t = 1 - self._beta[timestep]
        t_embedding = positional_embedding(timestep, d=32).repeat((1,32,1))

        x_t = torch.sqrt(alpha_bar_t)*x_0 + torch.sqrt(1-alpha_bar_t)*eps

        return x_0, x_t, eps, timestep, alpha_t, alpha_bar_t, t_embedding


class VocalFolds(torch.utils.data.Dataset):
    def __init__(
            self,
            split: str,
            timesteps: int = 1000,
            time_dims: int = 32,
            data_root: str = "./data",
    ) -> None:
        super().__init__()
        if split not in ["train_images", "val_images", "test_images"]:
            raise ValueError("Invalid split")
        self._split = split
        self._timesteps = timesteps
        self._time_dims = time_dims
        self._dataset_name = "vocalfolds"
        self._data_root = os.path.join(data_root, self._dataset_name)

        self.info = {
            "name": "vocalfolds",
            "url": "https://github.com/imesluh/vocalfolds.git",
        }

        filenames = self._download()

        # load data
        self._data = {
            "train_images": torch.stack(
                [torch.from_numpy(io.imread(f)).float() for f in filenames],  # .mean(dim=-1, keepdim=True)
                dim=0
            ).permute(0,3,1,2).float(),
            "val_images": [],
            "test_images": []
        }

        # variance schedule
        # self._alpha_bar = _compute_alpha_bar(self._timesteps)
        # self._beta = 1-(self._alpha_bar[1:]/self._alpha_bar[:-1])
        # self._beta = torch.cat((self._beta[0:1], self._beta)).clip(0, 0.999)
        self._beta = torch.linspace(1e-4, 0.02, self._timesteps)
        self._alpha = 1 - self._beta
        self._alpha_bar = torch.cumprod(self._alpha, dim=0)

        # preprocess data
        _preprocess_vmapped = torch.func.vmap(
            partial(_preprocess, size=(64, 64)),
            in_dims=0,
            out_dims=0
        )
        self._data[self._split] = _preprocess_vmapped(self._data[self._split])

    def _download(self):
        files = glob(os.path.join(self._data_root, "*.png"))
        if len(files) == 505:
            print("Vocalfolds already downloaded.")
            return files

        with TemporaryDirectory() as tmp_dir:
            subprocess.run([
                "git",
                "clone",
                self.info['url'],
                tmp_dir
            ])

            files = glob(os.path.join(tmp_dir, "img", "**", "**", "*.png"))
            os.makedirs(self._data_root, exist_ok=True)
            for file in files:
                copy2(
                    file,
                    os.path.join(self._data_root, os.path.basename(file))
                )
        
        return glob(os.path.join(self._data_root, "*.png"))

    def get_beta(self):
        return self._beta.clone()

    def get_alpha_bar(self):
        return self._alpha_bar.clone()

    def __len__(self):
        #return self._data[self._split].shape[0]
        return self._timesteps

    def __getitem__(self, param):
        if isinstance(param, int):
            t = param
            index = torch.randint(low=0, high=self._data[self._split].shape[0], size=(1,)).item()

            # t = torch.randint(low=0, high=self._timesteps, size=(1,))
            # t = (torch.distributions.Beta(1, 1.5).rsample()*self._timesteps).long()
        elif isinstance(param, tuple):
            t, index = param
        return self._get(index, t)

    def _get(self, index, timestep):
        x_0 = self._data[self._split][index]
        if torch.rand(1) > 0.5:
            x_0 = torch.flip(x_0, dims=(-1,))  # flip horizontally
        eps = torch.randn_like(x_0)

        alpha_bar_t = self._alpha_bar[timestep]
        alpha_t = 1 - self._beta[timestep]
        t_embedding = positional_embedding(timestep, d=self._time_dims)

        x_t = torch.sqrt(alpha_bar_t)*x_0 + torch.sqrt(1-alpha_bar_t)*eps

        return x_0, x_t, eps, timestep, alpha_t, alpha_bar_t, t_embedding
