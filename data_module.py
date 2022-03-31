import glob
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy
from torchvision.transforms import Resize, InterpolationMode


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, k_fold, kth_fold, dataset_path, config=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.k_fold = k_fold
        self.kth_fold = kth_fold
        self.dataset_path = dataset_path
        self.pin_memory = True
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None) -> None:
        k_fold_dataset_list = self.get_k_fold_dataset_list()
        if stage == 'fit' or stage is None:
            dataset_train, dataset_val = self.get_fit_dataset_lists(k_fold_dataset_list)
            self.train_dataset = CustomDataset(self.dataset_path, dataset_train, 'train', self.config, )
            self.val_dataset = CustomDataset(self.dataset_path, dataset_val, 'val', self.config, )
        if stage == 'test' or stage is None:
            dataset_test = self.get_test_dataset_lists(k_fold_dataset_list)
            self.test_dataset = CustomDataset(self.dataset_path, dataset_test, 'test', self.config, )

    def get_k_fold_dataset_list(self):
        # 得到用于K折分割的数据的list, 并生成文件夹进行保存
        if not os.path.exists(self.dataset_path + '/k_fold_dataset_list.txt'):
            # 获得用于k折分割的数据的list
            time_list = glob.glob(self.dataset_path + '/*')
            dataset = []
            for time in time_list:
                time = time.split('\\')[-1]
                dataset = dataset + glob.glob(
                    self.dataset_path + '/' + time + '/segmentation_mask_polar_' + time + '/*.npy')
            random.shuffle(dataset)
            written = dataset

            with open(self.dataset_path + '/k_fold_dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in written:
                    f.write(line.replace('\\', '/') + '\n')
            print('已生成新的k折数据list')
        else:
            dataset = open(self.dataset_path + '/k_fold_dataset_list.txt').readlines()
            dataset = [item.strip('\n') for item in dataset]
        return dataset

    def get_fit_dataset_lists(self, dataset_list: list):
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(len(dataset_list), self.k_fold)
        # 分割全部数据, 得到训练集, 验证集
        dataset_val = dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)]
        del (dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)])
        dataset_train = dataset_list
        return dataset_val, dataset_train

    def get_test_dataset_lists(self, dataset_list):
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(len(dataset_list), self.k_fold)
        # 分割全部数据, 得到测试集
        dataset = dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)]
        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        """
        由于pl计算验证epoch的loss的方法为每个batch的loss求均值, 而每个batch内计算多个样本时同样会求均值,
        这导致了两次求均值(官方的celoss也会出现该问题). 如果这时存在两个batch的size不同,则会导致每个loss
        的权重不相等, 导致求loss的错误. 这种情况常常出现, 因为数据量不能整除size, 所以基本上最后一个batch
        的size与前面的batch的size不同.
        为了保证验证集计算loss的准确性, 该方法中对于验证集的batch size进行了重新定义.
        此外, 训练集的反向传播不受影响, 但训练集的loss记录会受影响. 然而, 由于batch_size对训练结果具有较大
        的影响, 因此, 忽略loss记录的影响, 不对训练集batch size进行重新定义.
        """
        val_batch_size = 1
        for num in range(self.batch_size):
            if len(self.val_dataset) % (self.batch_size - num) == 0:
                val_batch_size = self.batch_size - num
                break
        return DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class CustomDataset(Dataset):
    def __init__(self, dataset_path, dataset, stage, config):
        super().__init__()
        self.dim = 3
        if config['model_name'] == 'unet2d':
            self.dim = 2
        self.dataset = dataset
        self.avg_pool_input = torch.nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.resize = Resize((128, 128), InterpolationMode.NEAREST)

    def __getitem__(self, idx):
        # 注意: 为了满足初始化权重算法的要求, 需要输入参数的均值为0. 可以使用transforms.Normalize()
        data_path = self.dataset[idx].replace('\\', '/')
        data_name = os.path.basename(data_path)
        # 角度*距离
        label_np = numpy.load(data_path)
        label_np = numpy.rot90(numpy.transpose(numpy.rot90(label_np, -1)[::-1]), 2)
        real_label = torch.from_numpy(label_np)
        label = real_label.unsqueeze(0)
        label = self.resize(label)
        label = label.squeeze(0)
        data_path = data_path.replace('segmentation_mask_polar', 'ral_outputs', ).split('/')
        data_path.insert(-1, 'RAD_numpy')
        data_path = '/'.join(data_path)
        data_np = numpy.load(data_path)
        mean = 98.93482236463528
        std = 838.2424861362456
        # max = 1339148.8
        data_np = (data_np - mean) / std
        data = torch.from_numpy(data_np)
        data = data.unsqueeze(0)
        data = self.avg_pool_input(data)
        if self.dim == 2:
            v_length = data.shape[3]
            data = torch.sum(data, dim=3) / v_length
        # 不需要输出image_name, 因此这里置为0. 如果置为str类型, 会导致错误的提示, 这是pl的BUG, 后续BUG解决后可以不使用0代替
        return 0, data, label,real_label

    def __len__(self):
        return int(len(self.dataset))
