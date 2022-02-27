import random

from torch.utils.tensorboard import SummaryWriter

from utils.plotting_utils import plot_alignment_to_numpy
from utils.plotting_utils import plot_spectrogram_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, mel_loss_l1, mel_loss_l2, dur_loss, grad_norm, duration,
                     iteration):
        self.add_scalar("Train/Total loss", reduced_loss, iteration)
        self.add_scalar("Train/Gradient norm", grad_norm, iteration)

        self.add_scalar("Train/L1 Mel Loss", mel_loss_l1, iteration)
        self.add_scalar("Train/L2 Mel Loss", mel_loss_l2, iteration)
        self.add_scalar("Train/Duration Loss", dur_loss, iteration)

    def log_training_loop(self, y, y_pred, iteration):
        _, mel_outputs, durations_pred, alignments = y_pred
        mel_targets, durations_target = y

        idx = random.randint(0, alignments.size(0) - 1)

        self.add_image("Train/alignment", plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                       iteration, dataformats='HWC')
        self.add_image("Train/mel_target", plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                       iteration, dataformats='HWC')
        self.add_image("Train/mel_predicted",
                       plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                       iteration, dataformats='HWC')

    def log_validation(self, reduced_loss, mel_loss_l1, mel_loss_l2, dur_loss, model, y, y_pred, iteration):

        self.add_scalar("Test/Total loss", reduced_loss, iteration)
        self.add_scalar("Test/L1 Mel Loss", mel_loss_l1, iteration)
        self.add_scalar("Test/L2 Mel Loss", mel_loss_l2, iteration)
        self.add_scalar("Test/Duration Loss", dur_loss, iteration)

        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        self.add_image("Test/alignment", plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                       iteration, dataformats='HWC')
        self.add_image("Test/mel_target", plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                       iteration, dataformats='HWC')
        self.add_image("Test/mel_predicted", plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                       iteration, dataformats='HWC')