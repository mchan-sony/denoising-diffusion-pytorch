import argparse
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transform
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion, UViT
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


class Cityscapes(Dataset):
    def __init__(self, image_size):
        self.files = glob(
            "/data/Cityscapes_processed/Cityscapes_h416/leftImg8bit/train/**/*.png"
        )
        self.tfm = transform.Compose(
            [
                transform.ToTensor(),
                transform.Resize([image_size, image_size], antialias=True),
                transform.RandomHorizontalFlip(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.tfm(cv2.imread(self.files[index]))
        return img


def train(device, config):
    dset = Cityscapes(config.image_size)
    loader = DataLoader(dset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    model = DataParallel(UViT(dim=64)).to(device)
    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint))
        print(f"Loaded model checkpoint from {config.checkpoint}!")
    model.train()
    diffusion = GaussianDiffusion(model, image_size=config.image_size)

    print(f"Starting training on {device} for {config.epochs} epochs!")
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    pbar = tqdm(range(config.epochs))
    losses = []
    for _ in pbar:
        for imgs in loader:
            optim.zero_grad()
            loss = diffusion(imgs.to(device))
            loss.backward()
            optim.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            losses.append(loss.item())
    losses = torch.from_numpy(np.array(losses))

    torch.save(model.state_dict(), os.path.join(config.write_dir, "model.pt"))
    torch.save(losses, os.path.join(config.write_dir, "loss.pt"))

    config.checkpoint = os.path.join(config.write_dir, "model.pt")


def sample(device, config):
    print("Sampling images...")
    model = UViT(dim=64).to(device)
    model.load_state_dict(torch.load(config.checkpoint))
    model.eval()
    diffusion = GaussianDiffusion(model, image_size=config.image_size)
    samples = diffusion.sample(batch_size=8)
    save_image(
        make_grid(samples.detach().cpu()), os.path.join(config.write_dir, "samples.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--write_dir", default="out", type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    config = parser.parse_args()
    print(config)

    write_dir = Path(config.write_dir).mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.train:
        train(device, config)
    if config.sample:
        assert not config.checkpoint is None, "No model checkpoint provided!"
        sample(device, config)
