# image_enhancement/image_enhancement/trainers/cnn_trainer.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models import UNet
from ..losses import SSIML1PerceptualLoss
from ..datasets import PairedDataset, get_train_transforms, get_val_transforms
from ..utils import setup_logger, get_device, save_checkpoint  # 导入工具函数
import csv
from datetime import datetime


class CNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device(config["device"])  # 使用工具函数获取设备

        # 1. 初始化模型、损失函数、优化器
        self.model = UNet(
            n_channels=config.get("n_channels", 3),
            n_classes=config.get("n_classes", 3),
            bilinear=config.get("bilinear", False)
        ).to(self.device)
        self.criterion = SSIML1PerceptualLoss(
            lambda_l1=config["lambda_l1"],
            lambda_ssim=config["lambda_ssim"],
            lambda_perceptual=config["lambda_perceptual"]
        ).to(self.device)
        betas = tuple(map(float, config["betas"]))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            betas=betas # type: ignore
        )

        # 2. 初始化数据加载器
        train_dataset = PairedDataset(
            data_root=config["data_root"],
            phase="train",
            transforms=get_train_transforms(config["image_size"])
        )
        val_dataset = PairedDataset(
            data_root=config["data_root"],
            phase="val",
            transforms=get_val_transforms(config["image_size"])
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"]
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )

        # 3. 初始化日志和保存目录（使用工具函数）
        self.save_dir = os.path.join(config["save_dir"], config["exp_name"])
        self.logger = setup_logger(self.save_dir)  # 初始化日志
        self.best_val_loss = float("inf")
        self.csv_log_path = os.path.join(self.save_dir, f"train_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        csv_header = ["epoch", "train_loss", "train_time(s)", "val_loss", "val_time(s)"]
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        self.logger.info(f"CSV log file initialized at: {self.csv_log_path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, batch in pbar:
            raw = batch["raw"].to(self.device)
            gt = batch["gt"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(raw)
            loss, loss_dict = self.criterion(pred, gt)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % self.config["log_interval"] == 0:
                log_str = f"Train Epoch: {epoch} [{i + 1}/{len(self.train_loader)}] " \
                          f"Loss: {loss.item():.4f} (L1: {loss_dict['l1']:.4f}, " \
                          f"SSIM: {loss_dict['ssim']:.4f}, Perceptual: {loss_dict['perceptual']:.4f})"
                pbar.set_description(log_str)
                self.logger.info(log_str)  # 使用logger记录

        avg_loss = total_loss / len(self.train_loader)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Train Epoch {epoch} finished. Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss, elapsed_time

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for i, batch in pbar:
                raw = batch["raw"].to(self.device)
                gt = batch["gt"].to(self.device)

                pred = self.model(raw)
                loss, _ = self.criterion(pred, gt)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Validation Epoch {epoch} finished. Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss, elapsed_time

    def save_checkpoint(self, epoch, is_best=False):
        # 使用工具函数保存检查点
        save_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            best_loss=self.best_val_loss,
            save_dir=self.save_dir,
            is_best=is_best
        )
        if is_best:
            self.logger.info(f"New best model saved to {save_path} (epoch {epoch})")

    def run(self):
        self.logger.info("Starting training...")
        for epoch in range(1, self.config["epochs"] + 1):
            train_loss, train_time = self.train_epoch(epoch)
            if epoch % self.config["val_interval"] == 0:
                val_loss, val_time = self.validate(epoch)
                with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, train_time, val_loss, val_time])

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            # 每10轮保存一次检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
        self.logger.info(f"Training finished! Best val loss: {self.best_val_loss:.4f}")
        self.logger.info(f"All logs saved at: {self.save_dir}")