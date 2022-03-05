from torch import nn


class NATLoss(nn.Module):
    def __init__(self):
        super(NATLoss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, duration_target = targets[0], targets[1]
        mel_target.requires_grad = False
        duration_target.requires_grad = False

        mel_out, mel_out_postnet, duration_pred, _ = model_output
        mel_loss_l1 = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        mel_loss_l2 = nn.L1Loss()(mel_out, mel_target) + \
            nn.L1Loss()(mel_out, mel_target)
        mel_loss = mel_loss_l1 + mel_loss_l2

        dur_loss = nn.MSELoss()(duration_pred, duration_target)

        print("mel_loss_l1", mel_loss_l1.item())
        print("mel_loss_l2", mel_loss_l2.item())
        print("dur_loss", dur_loss.item())

        return mel_loss + 2 * dur_loss, mel_loss_l1, mel_loss_l2, dur_loss
