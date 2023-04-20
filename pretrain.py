import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import ClassDataLoader, ClassMemDataLoader
from train import define_model, train_epoch, validate
from condense import load_resized_data, diffaug


def main(args, logger, repeat=1):
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)
    nclass = trainset.nclass
    _, aug_rand = diffaug(args)

    for i in range(repeat):
        logger(f"\nRepeat: {i + 1}/{repeat}")
        model = define_model(args, nclass, logger)
        model.train()
        pretrain(model, loader_real, val_loader, aug_rand, logger, i, args)


def pretrain(model, loader_real, val_loader, aug_rand, logger, args):
    criterion = nn.CrossEntropyLoss().cuda()
    optim_net = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    model = model.cuda()
    epoch_max = args.pt_from + args.pt_num  # pt_from=-1, pt_num=1
    scheduler = optim.lr_scheduler.MultiStepLR(
        optim_net, milestones=[2 * epoch_max // 3, 5 * epoch_max // 6], gamma=0.2)
    print(f"Start training for {epoch_max} epochs")
    best_acc = 0
    for epoch in range(1, epoch_max):
        top1, _, loss = train_epoch(args,
                                    loader_real,
                                    model,
                                    criterion,
                                    optim_net,
                                    epoch=epoch,
                                    aug=aug_rand,
                                    mixup=args.mixup_net,
                                    logger=logger)
        top1_val, _, _ = validate(args, val_loader, model, criterion, epoch, logger=logger)
        print(f"[Epoch {epoch}] Train acc: {top1:.1f} (loss: {loss:.3f}), Val acc: {top1_val:.1f}")
        scheduler.step()

        if epoch == 10 or epoch == 50 or epoch ==100:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_' + str(epoch) + '.pth.tar')
            torch.save(model.state_dict(), ckpt_path)

        if top1_val > best_acc:
            best_acc = top1_val
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_best' + '.pth.tar')
            torch.save(model.state_dict(), ckpt_path)

    logger(f"Best Accuracy: {best_acc}")


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn

    assert args.pt_from > 0, "set args.pt_from positive! (epochs for pretraining)"

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = f"./pretrained_models/{args.datatag}/{args.modeltag}{args.tag}_seed_{args.seed}_lr_{args.lr}_aug_{args.aug_type}"
    os.makedirs(args.save_dir, exist_ok=True)

    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    logger(f"Seed: {args.seed}")
    logger(f"Lr: {args.lr}")
    logger(f"Aug-type: {args.aug_type}")
    main(args, logger, args.repeat)