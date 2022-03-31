import torch


def calculate_acc(hist):
    """
    计算准确率, 而不是iou

    :param hist:
    :return:
    """
    n_class = hist.size()[0]
    conf = torch.zeros((n_class, n_class))
    for cid in range(n_class):
        if torch.sum(hist[:, cid]) > 0:
            conf[:, cid] = hist[:, cid] / torch.sum(hist[:, cid])

    # 可以看作对于除了背景外的像素点的判断accuracy, 但是比较偏向于判断为某些类的正确率.
    # nan表示均判断为背景, 如果存在除背景外的类别, 则正确率为0; 如果不存在, 则表示nan(无结果,若不去除背景,则正确率为1)
    overall_acc = torch.sum(torch.diag(hist[1:, 1:])) / torch.sum(hist[1:, :])

    # acc为某类预测结果是正确的概率
    # nan表示无像素判断为该类, 若存在该类, 则表示正确率为0; 若不存在, 则表示nan(无法判断为该类结果的正确率)
    acc = torch.diag(conf)

    return overall_acc, acc
