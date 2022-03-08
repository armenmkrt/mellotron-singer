import argparse
import os
import time
from pathlib import Path

import torch
from numpy import finfo
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.helper_funcs as helpers
from dataset import TextMelDurCollate
from dataset import TextMelDurLoader
from loss import NATLoss
from modules.nn import NAT
from utils.distributed import apply_gradient_allreduce
from utils.param_utils import HParams

os.chdir(Path(os.getcwd()).parent)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(42)


def prepare_dataloaders(hparams_):
    # Get data, data loaders and collate function ready
    trainset = TextMelDurLoader(os.path.join(hparams_.dataset_path, 'train.txt'), hparams_)
    valset = TextMelDurLoader(os.path.join(hparams_.dataset_path, 'val.txt'), hparams_)
    collate_fn = TextMelDurCollate(hparams_.n_frames_per_step,
                                   trainset.durations_length_mean,
                                   trainset.durations_length_std)

    if hparams_.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=10, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams_.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def load_model(hparams_):
    model = NAT(hparams_).cuda()
    if hparams_.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams_.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn,
                                drop_last=True)

        val_loss = 0.0
        val_mel_loss_l1 = 0.0
        val_mel_loss_l2 = 0.0
        val_dur_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss, mel_loss_l1, mel_loss_l2, dur_loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = helpers.reduce_tensor(loss.data, n_gpus).item()
                mel_loss_l1 = 0
                mel_loss_l2 = 0
                dur_loss = 0
            else:
                reduced_val_loss = loss.item()
                mel_loss_l1 = mel_loss_l1.item()
                mel_loss_l2 = mel_loss_l2.item()
                dur_loss = dur_loss.item()

            val_loss += reduced_val_loss
            val_mel_loss_l1 += mel_loss_l1
            val_mel_loss_l2 += mel_loss_l2
            val_dur_loss += dur_loss

        val_loss = val_loss / (i + 1)
        val_mel_loss_l1 = val_mel_loss_l1 / (i + 1)
        val_mel_loss_l2 = val_mel_loss_l2 / (i + 1)
        val_dur_loss = val_dur_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, val_mel_loss_l1, val_mel_loss_l2, val_dur_loss,
                              model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams_):

    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams_.distributed_run:
        helpers.init_distributed(hparams_, n_gpus, rank, group_name)

    torch.manual_seed(hparams_.seed)
    torch.cuda.manual_seed(hparams_.seed)

    model = load_model(hparams_)
    learning_rate = hparams_.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if hparams_.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = NATLoss()

    logger = helpers.prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams_)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = helpers.warm_start_model(
                checkpoint_path, model, hparams_.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = helpers.load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams_.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            print("epoch_offset", epoch_offset)

    model.train()
    is_overflow = False
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams_.epochs):
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            # try:
            y_pred = model(x)
            loss, mel_loss_l1, mel_loss_l2, dur_loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            mel_loss_l1 = mel_loss_l1.item()
            mel_loss_l2 = mel_loss_l2.item()
            dur_loss = dur_loss.item()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams_.grad_clip_thresh)
            optimizer.step()

            if iteration % 10 == 0:
                logger.log_training_loop(y, y_pred, iteration)

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Epoch {} Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                      epoch, iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, mel_loss_l1, mel_loss_l2, dur_loss,
                    grad_norm, duration, iteration)

            if not is_overflow and (iteration % hparams_.iters_per_checkpoint == 0):
                print("validation run")
                validate(model, criterion, valset, iteration,
                         1, n_gpus, collate_fn, logger,
                         hparams_.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    helpers.save_checkpoint(model, optimizer, learning_rate, iteration,
                                            checkpoint_path)

            if iteration == hparams_.stop_iteration:
                break
            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str,
                        default="/home/podcastle/Documents/mellotron-singer/configs/train.yaml",
                        help="hyperparameters config file path")
    parser.add_argument('-o', '--output_directory', type=str,
                        default="/home/podcastle/Documents/mellotron-singer/models/base_model/",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default="./runs/",
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')

    args = parser.parse_args()
    print(args.checkpoint_path)
    hparams = HParams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
