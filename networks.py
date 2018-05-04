import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        #self.batch_size = batch_size
        self.l1 = nn.Sequential(
            nn.Linear(nz, 256*7*7))
        self.l2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=5, padding=2, stride=1),
            nn.Tanh())
    def forward(self, z):
        out = self.l1(z)
        out = out.view(z.size(0),256,7,7)
        out = self.l2(out)
        return out

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.D1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.02))

        self.D_disc = nn.Linear(256*7*7,1)

    def compute_h(self,x):
        #x = x.resize(x.size(0), 3, 32, 32)


        h = self.D1(x)


        h = h.resize(x.size(0), 256*7*7)

        return h

    def forward(self,x):

        h = self.compute_h(x)

        y_disc = self.D_disc(h)

        return y_disc

