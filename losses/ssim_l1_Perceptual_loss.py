import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchmetrics import StructuralSimilarityIndexMeasure


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_relu1_1 = self.slice1(x)
        y_relu1_1 = self.slice1(y)
        loss = F.l1_loss(x_relu1_1, y_relu1_1)
        return loss


class SSIML1PerceptualLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual

        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = 1 - self.ssim(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        total_loss = (self.lambda_l1 * l1 +
                      self.lambda_ssim * ssim +
                      self.lambda_perceptual * perceptual)
        return total_loss, {"l1": l1.item(), "ssim": ssim.item(), "perceptual": perceptual.item()}