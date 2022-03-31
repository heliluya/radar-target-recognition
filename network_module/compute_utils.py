import torch


def one_hot_encoder(input_tensor, n_classes):
    """
    将输入tensor转化为one-hot形式

    :param input_tensor:
    :param n_classes:
    :return:
    """
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.long()


def torch_nanmean(x):
    """
    输出忽略nan的tensor均值

    :param x:
    :return:
    """
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    # num为0表示均为nan, 此时由于分母不能为0, 则设num为1
    if num == 0:
        num = 1
    return value / num
