import torchvision.transforms as transforms

# CIFAR-100 のチャネル平均・標準偏差
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def pretrain_train_transform():
    """事前学習用。32×32 の RandomCrop（padding 4）と左右反転。"""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def cifar100_eval_transform():
    """評価用、および少数サンプルのファインチューニング用（拡張なし）。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
