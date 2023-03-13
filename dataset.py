import os
import numpy as np
import torch
from functorch import vmap


def positional_embedding(t, d=32):
    """
    Transformer positional embedding.
    """
    k = torch.arange(0, d//2, 1)
    w = torch.exp(k*-np.log(10000)/d)

    p = torch.empty((d,))
    p[0::2] = torch.sin(t*w)
    p[1::2] = torch.cos(t*w)
    return p


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
            "url": "https://zenodo.org/record/6496656/files/chestmnist.npz",
            "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        }

        self._download()

        self._data = {}
        dataset = np.load(os.path.join(self._data_root, f"{self.info['name']}.npz"))
        for key in dataset.files:
            self._data[key] = torch.from_numpy(dataset[key]).unsqueeze(1)

        # variance schedule
        self._alpha_bar = self._compute_alpha_bar(self._timesteps)
        self._beta = 1-(self._alpha_bar[1:]/self._alpha_bar[:-1])
        self._beta = torch.cat((self._beta[0:1], self._beta)).clip(0, 0.999)

        # preprocess data
        _preprocess_vmapped = vmap(self._preprocess, in_dims=0, out_dims=0)
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

    @staticmethod
    def _preprocess(img):
        img = img.float()*2/255 - 1  # scale to [-1, 1]
        # img = torch.nn.functional.pad(img[None,], pad=(2,2,2,2), mode="reflect")[0]
        img = torch.nn.functional.interpolate(img[None,], size=(32, 32), mode="bilinear")[0]
        return img

    @staticmethod
    def _compute_alpha_bar(timesteps: int, s: float = 0.008):
        t = torch.arange(timesteps)
        return torch.cos((t/timesteps+s)/(1+s)*np.pi/2)**2 / torch.cos((torch.tensor([0])+s)/(1+s)*np.pi/2)**2

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
