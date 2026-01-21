import torch
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

def load_yolo(repo, weights):
    return torch.hub.load(
        repo,
        "custom",
        path=weights,
        source="local"
    )
