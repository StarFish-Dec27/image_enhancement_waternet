# image_enhancement/image_enhancement/utils/utils.py
import os
import logging
import torch


def setup_logger(save_dir, log_name="train.log"):
    """初始化日志记录器"""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)


def get_device(prefer="cuda"):
    """获取可用的计算设备"""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_loss = checkpoint.get("best_val_loss", float("inf"))
    epoch = checkpoint.get("epoch", 0)
    return model, optimizer, best_loss, epoch


def save_checkpoint(model, optimizer, epoch, best_loss, save_dir, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_loss
    }
    if is_best:
        save_path = os.path.join(save_dir, "best_model.pth")
    else:
        save_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    return save_path