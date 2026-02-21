from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        self.data_root = data_root
        self.transforms = transforms
        self.raw_list = []  # 待填充原始图像路径列表
        self.gt_list = []   # 待填充真值图像路径列表

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):
        raise NotImplementedError