from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip, Resize

# 定义训练数据的变换函数
def get_train_transforms(image_size):
    return Compose([
        Resize((image_size, image_size)),
        RandomCrop(image_size),  # 随机裁剪到指定尺寸
        RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        ToTensor(),  # 转换为Tensor格式
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

def get_val_transforms(image_size):
    return Compose([
        Resize((image_size, image_size)),# 调整到指定尺寸
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])