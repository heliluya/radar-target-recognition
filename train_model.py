import importlib
import time
import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.transforms import Resize, InterpolationMode

from network_module import iou
from network_module.dice_loss import DiceLoss


class TrainModule(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.time_sum = None
        self.config = config
        imported = importlib.import_module('network.%(model_name)s' % config)
        self.net = imported.Net(n_channels=1, n_classes=2, bilinear=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(self.config['n_classes'])
        self.resize = Resize((256, 256), InterpolationMode.NEAREST)

    # 返回值必须包含loss, loss可以作为dict中的key, 或者直接返回loss
    def training_step(self, batch, batch_idx):
        _, input, label, real_label = batch
        pred = self.net(input)
        loss_ce = self.ce_loss(pred, label.long())
        loss_dice = self.dice_loss(pred, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log("Training loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, input, label, real_label = batch
        pred = self.net(input)
        loss_ce = self.ce_loss(pred, label.long())
        loss_dice = self.dice_loss(pred, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log("Validation loss", loss)
        real_pred = self.resize(pred)
        miou, _ = iou.get_ious(real_pred, real_label, self.config['n_classes'], softmax=True)
        self.log("Validation acc", miou)
        return loss

    def test_step(self, batch, batch_idx):
        _, input, label = batch
        if self.time_sum is None:
            time_start = time.time()
            pred = self.net(input)
            time_end = time.time()
            self.time_sum = time_end - time_start
            print(f'\n推理时间为: {self.time_sum:f}')
        else:
            pred = self.net(input)
        loss_ce = self.ce_loss(pred, label.long())
        loss_dice = self.dice_loss(pred, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log("Test loss", loss)
        miou, _ = iou.get_ious(pred, label, self.config['n_classes'], softmax=True)
        self.log("Test acc", miou)
        return label, pred, miou

    def configure_optimizers(self):
        lr = 0.1
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
        # 仅在第一个epoch使用0.01的学习率
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr * 0.1
        return optimizer#, [lr_scheduler]

    def load_pretrain_parameters(self):
        """
        载入预训练参数
        """
        pass
