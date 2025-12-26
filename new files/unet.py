import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.d1 = block(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = block(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = block(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.bridge = block(256, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = block(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = block(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = block(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        b = self.bridge(self.p3(d3))

        u3 = self.c3(torch.cat([self.u3(b), d3], dim=1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], dim=1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], dim=1))

        return self.out(u1)
