# coding=utf-8
import os
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset


def get_dataloader_cifar(args):
    if args.dataset in ('domainnet', 'terra_incognita', 'digits_dg', 'dg5', 'VLCS'):
        rate = 0.2
    else:
        rate = 0.1

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    image_train = imgutil.image_train
    image_test = imgutil.image_test

    # Split only the original training images; keep the original test set unused.
    source_dir = os.path.join(args.data_dir, names[0], 'train')
    tmp_labels = ImageDataset(args.dataset, args.task, source_dir,
                              '', 0, transform=image_train(args), test_envs=args.test_envs).labels
    l = len(tmp_labels)
    if args.split_style == 'strat':
        lslist = np.arange(l)
        stsplit = ms.StratifiedShuffleSplit(
            2, test_size=rate, train_size=1 - rate,
            random_state=args.seed)
        stsplit.get_n_splits(lslist, tmp_labels)
        indextr, indexte = next(stsplit.split(lslist, tmp_labels))
    else:
        indexall = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(indexall)
        ted = int(l * rate)
        indextr, indexte = indexall[:-ted], indexall[-ted:]

    train_dataset = ImageDataset(args.dataset, args.task, source_dir,
                                 '', 0, transform=image_train(args), indices=indextr, test_envs=args.test_envs,
                                 shots_per_class=args.shots_per_class, num_augments=args.num_augments,
                                 ori_transform=image_test(args))
    val_dataset = ImageDataset(args.dataset, args.task, source_dir,
                               '', 0, transform=image_test(args), indices=indexte, test_envs=args.test_envs)

    train_loader = [DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=True,
        shuffle=True
    )]

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=100,
        num_workers=2 * args.N_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    # Build target_loaders from the corruptions folder (names[1]) — keep existing behaviour
    target_loaders = {}
    if len(names) > 1:
        target_dir = os.path.join(args.data_dir, names[1])
        if os.path.isdir(target_dir):
            for corruption in os.listdir(target_dir):
                corruption_path = os.path.join(target_dir, corruption)
                if os.path.isdir(corruption_path):
                    target_dataset = ImageDataset(args.dataset, args.task, corruption_path, domain_name="",
                                                  transform=image_test(args))
                    if target_dataset.classes != val_dataset.classes:
                        raise ValueError(f'CIFAR-100 class order differs in target domain: {corruption}')
                    target_loaders[corruption] = DataLoader(
                        dataset=target_dataset,
                        batch_size=100,
                        num_workers=2 * args.N_WORKERS,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False
                    )

    return train_loader, val_loader, target_loaders
