import os
import time
import logging
import csv
import torch
import numpy as np
from datetime import datetime
import yaml
import argparse
from image_enhancement.trainers import CNNTrainer


def setup_logging():
    # 创建日志目录
    log_dir = "./training_logs"
    ckpt_dir = "./checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 配置日志格式（同时输出到控制台+文件）
    log_filename = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_dir, ckpt_dir, log_filename

def parse_args():
    parser = argparse.ArgumentParser(description="Image Enhancement Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (.yaml)")
    return parser.parse_args()


def main():

    log_dir, ckpt_dir, log_filename = setup_logging()
    logging.info("Logging system initialized! Log file: {}".format(log_filename))

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = log_dir
    config["ckpt_dir"] = ckpt_dir

    trainer = CNNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()