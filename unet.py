import torch
import segmentation_models_pytorch as smp
import numpy as np

def load_unet(path):
    """
    Load trained U-Net model (2 classes: background + vegetation)
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2   # must match training
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def prepare(img, size=512):
    """
    Preprocess PIL image for U-Net inference
    """
    # Resize to training size
    img = img.resize((size, size))
    img_array = np.array(img) / 255.0

    # Convert to tensor: (C,H,W) and batch dimension
    img_tensor = torch.tensor(img_array).permute(2,0,1).unsqueeze(0).float()
    return img_tensor

def postprocess(mask):
    """
    Convert model output to a visual RGB mask for Streamlit
    """
    # mask: numpy array with 0=background, 1=vegetation
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[mask==1] = [0,255,0]  # green for vegetation
    return rgb_mask
