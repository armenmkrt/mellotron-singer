import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from librosa import load
from scipy.io.wavfile import read

from utils.logger import Tacotron2Logger


class ParseFromConfigFile(argparse.Action):

    def __init__(self, option_strings, type, dest, help=None, required=False):
        super(ParseFromConfigFile, self).__init__(option_strings=option_strings, type=type, dest=dest, help=help,
                                                  required=required)

    def __call__(self, parser, namespace, values, option_string):
        with open(values, 'r') as f:
            data = json.load(f)

        for group in data.keys():
            for k, v in data[group].items():
                underscore_k = k.replace('-', '_')
                setattr(namespace, underscore_k, v)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_torch_with_librosa(full_path, sr):
    data, sampling_rate = load(full_path, sr=sr)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_file_paths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split_and_upload_to_s3(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            return path, text

        file_paths_and_text = [split_line(dataset_path, line) for line in f]
    return file_paths_and_text


def load_file_paths_dur_and_phonemes(filename, splitter="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(splitter) for line in f] # added strip
    return filepaths_and_text


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def get_mask_from_lengths_nat(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def log_model_weights_distributions(model, summary_writer, steps, save_period):
    if steps % save_period == 0:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            summary_writer.add_histogram(tag, value.data.cpu().numpy(), steps)


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)



