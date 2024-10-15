import os
import pprint
import argparse
import torch.backends.cudnn as cudnn

from torch.cuda.amp import autocast, GradScaler


import torch
import torch.nn as nn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
import models as models
from models.wresnet1024 import ASTNet as get_net1
from models.wresnet2048 import ASTNet as get_net2
import datasets

from torch.cuda.amp import autocast, GradScaler
import gc


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = get_net2(config)
    model = model.cuda()

    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    losses = loss_util.MultiLossFunction(config=config).cuda()
    optimizer = optimizer_util.get_optimizer(config, model)
    scheduler = optimizer_util.get_scheduler(config, optimizer)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    losses = loss_util.MultiLossFunction(config=config).cuda()
    optimizer = optimizer_util.get_optimizer(config, model)
    scheduler = optimizer_util.get_scheduler(config, optimizer)

    
    train_dataset = eval('datasets.get_data')(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number videos: {}'.format(len(train_dataset)))

    last_epoch = config.TRAIN.BEGIN_EPOCH
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_loader, model, losses, optimizer, epoch, logger)

        scheduler.step()

        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            logger.info('=> saving model state epoch_{}.pth to {}\n'.format(epoch+1, final_output_dir))
            torch.save(model.module.state_dict(), os.path.join(final_output_dir,
                                                               'epoch_{}.pth'.format(epoch + 1)))
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger):
    loss_func_mse = nn.MSELoss(reduction='none')
    model.train()
    scaler = GradScaler()

    accumulation_steps = 4  # Adjust this value as needed


    for i, data in enumerate(train_loader):
        inputs, target = train_util.decode_input(input=data, train=True)
        target = target.cuda(non_blocking=True)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        with autocast():
            output = model(inputs)
            inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
            loss = inte_loss + grad_loss + msssim_loss + l2_loss

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Compute PSNR
        with torch.no_grad():
            mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
            psnr = anomaly_util.psnr_park(mse_imgs)

        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f}]\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=optimizer.param_groups[0]['lr'],
                                             inte=inte_loss.item(), grad=grad_loss.item(), 
                                             msssim=msssim_loss.item(), l2=l2_loss.item(),
                                             psnr=psnr)
            logger.info(msg)

        # Clear unnecessary tensors
        del inputs, target, output, loss
        torch.cuda.empty_cache()

    # Final optimization step for remaining accumulated gradients
    if len(train_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    main()