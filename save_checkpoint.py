import os
import numpy.random
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import shutil
from pytorch_lightning.utilities import rank_zero_info
from utils import zip_dir, fill_list
import re


class SaveCheckpoint(ModelCheckpoint):
    def __init__(self,
                 max_epochs,
                 config,
                 seed=None,
                 every_n_epochs=None,
                 path_final_save=None,
                 monitor=None,
                 save_top_k=None,
                 verbose=False,
                 mode='min',
                 no_save_before_epoch=0,
                 version_info='无',
                 save_last=False):
        """
        通过回调实现checkpoint的保存逻辑, 同时具有回调函数中定义on_validation_end等功能.

        :param max_epochs:
        :param seed:
        :param every_n_epochs:
        :param path_final_save:
        :param monitor:
        :param save_top_k:
        :param verbose:
        :param mode:
        :param no_save_before_epoch:
        :param version_info:
        """
        super().__init__(every_n_epochs=every_n_epochs, verbose=verbose, mode=mode, monitor=monitor,
                         save_top_k=save_top_k, save_last=save_last)
        self.path_final_save = path_final_save
        self.no_save_before_epoch = no_save_before_epoch
        self.version_info = version_info+','+config['version_info']
        self.seed = seed
        if seed is not None:
            numpy.random.seed(seed)
            self.seeds = numpy.random.randint(0, 2000, max_epochs)
            pl.seed_everything(seed)
            self.flag_sanity_check = 0

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        修改随机数逻辑,网络的随机种子给定,取样本的随机种子由给定的随机种子生成,保证即使重载训练每个epoch具有不同的抽样序列.
        同时保存checkpoint.

        :param trainer:
        :param pl_module:
        :return:
        """
        # 第一个epoch使用原始输入seed作为种子, 后续的epoch使用seeds中的第epoch-1个作为种子
        if self.seed is not None:
            if self.flag_sanity_check == 0:
                self.flag_sanity_check = 1
            else:
                pl.seed_everything(self.seeds[trainer.current_epoch])
        super().on_validation_end(trainer, pl_module)

    def _save_top_k_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates) -> None:
        epoch = monitor_candidates.get("epoch")
        version_name = self.get_version_name(self.dirpath)
        if self.monitor is None or self.save_top_k == 0 or epoch < self.no_save_before_epoch:
            return

        current = monitor_candidates.get(self.monitor)

        if self.check_monitor_top_k(trainer, current):
            self._update_best_and_save(current, trainer, monitor_candidates)

            if self.mode == 'max':
                saved_value = max([float('%.2f' % item) for item in list(self.best_k_models.values())])
            else:
                saved_value = min([float('%.2f' % item) for item in list(self.best_k_models.values())])
            self.save_version_info(version_name, epoch, saved_value)

            # 每次更新ckpt文件后, 将其存放到另一个位置
            if self.path_final_save is not None:
                zip_dir('./logs/default/' + version_name, './' + version_name + '.zip')
                if os.path.exists(self.path_final_save + '/' + version_name + '.zip'):
                    os.remove(self.path_final_save + '/' + version_name + '.zip')
                shutil.move('./' + version_name + '.zip', self.path_final_save)
        elif self.verbose:
            step = monitor_candidates.get("step")
            best_model_values = 'now best model:'
            for cou_best_model in self.best_k_models:
                best_model_values = ' '.join(
                    (best_model_values, str(round(float(self.best_k_models[cou_best_model]), 4))))
            rank_zero_info(
                f"\nEpoch {epoch:d}, global step {step:d}: {self.monitor} ({float(current):f}) was not in "
                f"top {self.save_top_k:d}({best_model_values:s})")

    # epoch为0表示未记录epoch或确实为0, epoch为-1表示这是测试阶段产生的结果
    def save_version_info(self, version_name, epoch, saved_value):
        # 版本信息表格的属性有: 版本名, epoch, 评价结果, 备注
        # 新增的话修改此处
        saved_info = [version_name, str(epoch), str(saved_value), self.version_info]
        # 保存版本信息(准确率等)到txt中
        if not os.path.exists('./logs/default/version_info.txt'):
            with open('./logs/default/version_info.txt', 'w', encoding='utf-8') as f:
                f.write(" ".join(saved_info) + '\n')
        else:
            with open('./logs/default/version_info.txt', 'r', encoding='utf-8') as f:
                info_list = f.readlines()
            info_list = [item.strip('\n').split(' ') for item in info_list]
            if len(info_list[0]) < len(saved_info):
                for cou in range(len(info_list)):
                    info_list[cou] = fill_list(info_list[cou], len(saved_info))
            else:
                saved_info = fill_list(saved_info, len(info_list[0]))
            # 对list进行转置, 转置后行为不同属性, 列为不同版本
            info_list = list(map(list, zip(*info_list)))
            if version_name in info_list[0]:
                for cou in range(len(info_list[0])):
                    if version_name == info_list[0][cou]:
                        for cou_attr in range(1, len(saved_info)):
                            info_list[cou_attr][cou] = saved_info[cou_attr]
            else:
                for cou_attr in range(len(saved_info)):
                    info_list[cou_attr].append(saved_info[cou_attr])
            # 对list进行转置
            info_list = list(map(list, zip(*info_list)))
            with open('./logs/default/version_info.txt', 'w', encoding='utf-8') as f:
                for line in info_list:
                    line = " ".join(line)
                    f.write(line + '\n')

    @staticmethod
    def get_version_name(dirpath):
        version_name = 'version_unkown'
        for item in re.split(r'[/|\\]', dirpath):
            if 'version_' in item:
                version_name = item
                break
        return version_name

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_version_info(self.get_version_name(trainer.log_dir), -1,
                               float('%.2f' % trainer.logged_metrics['Test acc']))
