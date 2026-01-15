from dataset import SegDataset

ds = SegDataset("images","masks","train.txt")
img, mask = ds[0]

print(img.shape)   # [3,H,W]
print(mask.shape)  # [H,W]
print(mask.unique())
