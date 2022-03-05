import json
import math
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from data.text import g2p_phonemes
from modules.layers import ConvNorm
from modules.layers import LinearNorm
from utils.helper_funcs import get_mask_from_lengths_nat
from utils.helper_funcs import to_gpu


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """
        Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class DurationPredictor(nn.Module):
    """
     Duration Predicting submodule
    """
    def __init__(self, hparams):
        super(DurationPredictor, self).__init__()
        self.input_size = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
        self.duration_rnn_dim = hparams.duration_rnn_dim
        self.duration_rnn_num_layers = hparams.duration_rnn_num_layers

        self.duration_rnn = nn.LSTM(input_size=self.input_size,
                                    hidden_size=self.duration_rnn_dim,
                                    batch_first=True, bidirectional=True,
                                    num_layers=self.duration_rnn_num_layers)

        self.duration_proj_layer = nn.Linear(in_features=2 * self.duration_rnn_dim,
                                             out_features=1)
        torch.nn.init.uniform_(self.duration_proj_layer.bias, -1, 1)

    def forward(self, encoder_outputs, input_lengths):
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(encoder_outputs,
                                              input_lengths,
                                              batch_first=True)

        self.duration_rnn.flatten_parameters()
        x, _ = self.duration_rnn(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = self.duration_proj_layer(x)

        return x

    def inference(self, encoder_outputs):
        self.duration_rnn.flatten_parameters()
        x, _ = self.duration_rnn(encoder_outputs)
        x = self.duration_proj_layer(x)

        return x


class RangePredictor(nn.Module):
    def __init__(self, hparams):
        super(RangePredictor, self).__init__()
        self.input_size = hparams.range_model_input_size
        self.range_rnn_dim = hparams.range_rnn_dim
        self.range_rnn_num_layers = hparams.range_rnn_num_layers

        print(self.input_size,
              self.range_rnn_dim,
              self.range_rnn_num_layers)

        self.range_rnn = nn.LSTM(input_size=self.input_size,
                                 hidden_size=self.range_rnn_dim,
                                 batch_first=True, bidirectional=True,
                                 num_layers=self.range_rnn_num_layers)

        self.range_proj_layer = nn.Linear(in_features=2 * self.range_rnn_dim,
                                          out_features=1)
        torch.nn.init.constant_(self.range_proj_layer.bias, 5)

        self.softplus = nn.Softplus()

    def forward(self, x, input_lengths):
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True)

        self.range_rnn.flatten_parameters()
        x, _ = self.range_rnn(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = self.range_proj_layer(x)
        x = self.softplus(x)
        return x

    def inference(self, x):
        self.range_rnn.flatten_parameters()
        x, _ = self.range_rnn(x)
        x = self.range_proj_layer(x)
        x = self.softplus(x)
        return x


class GaussianUpsampling(nn.Module):
    def __init__(self):
        super(GaussianUpsampling, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, durations, sigma, decoder_step):
        # Calculating c_i vectors
        c_vectors = torch.cumsum(durations, dim=1) - durations / 2
        c_vectors = c_vectors.unsqueeze(-1)

        sigma = sigma + 1e-5

        prob = 1 / torch.sqrt(2 * np.pi * sigma ** 2) * torch.exp(-((decoder_step - c_vectors) ** 2) / (sigma ** 2) / 2)
        prob += 1e-5

        # Getting weights

        denominator = torch.sum(prob, dim=1).unsqueeze(-1)
        weights = prob / denominator
        encoder_outputs = encoder_outputs.transpose(1, 2)
        weighted_encoder_outputs = torch.bmm(encoder_outputs, weights)
        weighted_encoder_outputs = weighted_encoder_outputs.squeeze(-1)
        return weighted_encoder_outputs, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, pos):
        pe_expanded = self.pe[pos]
        x = torch.cat([x, pe_expanded], -1)
        return x


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.first_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dim + hparams.positional_encoding_d,
            hparams.attention_rnn_dim)

        self.gaussian_upsampling = GaussianUpsampling()
        self.positional_encoding = PositionalEncoding(hparams.positional_encoding_d,
                                                      hparams.positional_encoding_max_len)

        self.second_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim,
            hparams.decoder_rnn_dim, True)

        self.linear_projection = LinearNorm(
            2048,  # 2048
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.first_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.first_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.second_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.second_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)

        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)

        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, alignments

    def decode(self, decoder_step, encoder_outputs, decoder_input, durations_in_frames, duration_frames_indices, range_pred):
        """
        Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        sampled_encoder_outputs, self.alignment_weights = self.gaussian_upsampling(encoder_outputs,
                                                                                   durations_in_frames,
                                                                                   range_pred,
                                                                                   decoder_step)
        positions = duration_frames_indices[:, decoder_step]
        sampled_encoder_outputs = self.positional_encoding(sampled_encoder_outputs, positions)

        first_input = torch.cat((decoder_input, sampled_encoder_outputs), -1)

        self.first_hidden, self.first_cell = self.first_rnn(first_input,
                                                            (self.first_hidden,
                                                             self.first_cell))

        self.first_hidden = F.dropout(self.first_hidden,
                                      self.p_decoder_dropout,
                                      self.training)

        self.second_hidden, self.second_cell = self.second_rnn(self.first_hidden,
                                                               (self.second_hidden,
                                                                self.second_cell))
        self.second_hidden = F.dropout(self.second_hidden,
                                       self.p_decoder_dropout,
                                       self.training)

        decoder_hidden_context = torch.cat((self.second_hidden, sampled_encoder_outputs), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_context)
        return decoder_output, self.alignment_weights

    def forward(self, memory, durations_in_frames, duration_frames_indices,
                range_pred, decoder_inputs,  memory_lengths):
        """
        Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths_nat(memory_lengths))

        mel_outputs, alignments = [], []
        decoder_step = 0
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, alignment_weights = self.decode(decoder_step, memory,
                                                        decoder_input, durations_in_frames,
                                                        duration_frames_indices, range_pred)
            mel_outputs += [mel_output.squeeze(1)]
            alignments += [alignment_weights]
            decoder_step += 1
        mel_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, alignments)
        alignments = alignments.squeeze(-1)

        return mel_outputs, alignments

    def inference(self, memory, durations_in_frames, unpacked_durations, range_pred):
        """
        Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, alignments = [], []
        decoder_step = 0
        for _ in unpacked_durations[0]:
            decoder_input = self.prenet(decoder_input)
            mel_output, alignment = self.decode(decoder_step, memory,
                                                decoder_input, durations_in_frames,
                                                unpacked_durations, range_pred)

            mel_outputs += [mel_output.squeeze(1)]
            alignments += [alignment]
            decoder_step += 1
            decoder_input = mel_output

        mel_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, alignments)

        return mel_outputs, alignments


class NAT(nn.Module):
    def __init__(self, hparams):
        """
        Non-Attentive Tacotron implementation
        https://arxiv.org/abs/2010.04301
        :param hparams: config file object
        """
        super(NAT, self).__init__()
        # Output mask padding (bool)
        self.mask_padding = hparams.mask_padding

        # Audio configs
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.dataset_path = hparams.dataset_path
        self.durations_length_mean, self.durations_length_std = self.get_duration_stats()

        # Embedding layer initialization
        n_symbols = len(g2p_phonemes)
        self.embedding = nn.Embedding(n_symbols,
                                      hparams.symbols_embedding_dim,
                                      padding_idx=0)
        std = math.sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Initializing submodules
        self.encoder = Encoder(hparams)
        self.duration_predictor = DurationPredictor(hparams)
        self.range_predictor = RangePredictor(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    @staticmethod
    def parse_batch(batch):
        """
        Assign model inputs to gpu
        :param batch:
        :return (tuple): tuple containing input tensors and pad information
        """

        phonemes_padded, durations_padded, unpacked_durations_padded, durations_in_frames, input_lengths, mel_padded, \
            output_lengths, embeds = batch
        phonemes_padded = to_gpu(phonemes_padded).long()
        durations_padded = to_gpu(durations_padded).float()
        unpacked_durations_padded = to_gpu(unpacked_durations_padded).long()
        durations_in_frames = to_gpu(durations_in_frames).long()

        max_len = torch.max(input_lengths.data).item()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        embeds = to_gpu(embeds).float()

        return ((phonemes_padded, unpacked_durations_padded, durations_in_frames,
                 input_lengths, mel_padded, max_len, output_lengths, embeds),
                (mel_padded, durations_padded))

    def parse_output(self, outputs, mel_output_lengths=None, dur_outputs_lengths=None) -> list:
        """
        Masks mel-spectrograms, alignment and durations with corresponding values
        :param outputs: list containing mel, postnet mel, alignmetn and duration
        :param mel_output_lengths: lengths of mel in batch
        :param dur_outputs_lengths: lengths of durations in batch
        :return: masked outputs of the model
        """
        if self.mask_padding and mel_output_lengths is not None:
            # Mel masking
            mel_mask = ~get_mask_from_lengths_nat(mel_output_lengths)
            mel_mask = mel_mask.expand(self.n_mel_channels, mel_mask.size(0), mel_mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mel_mask, 0.0)
            outputs[1].data.masked_fill_(mel_mask, 0.0)

            # Duration masking
            dur_mask = ~get_mask_from_lengths_nat(dur_outputs_lengths)
            outputs[2].data.masked_fill_(dur_mask, (0 - self.durations_length_mean) / self.durations_length_std)  # bug here

        return outputs

    def forward(self, inputs):
        """
        Performs model forward in train mode
        :param inputs: batch assigned to gpu packed in tuple
        :return: model outputs masked by parse_output method
        """
        text_inputs, duration_frames_indices, durations_in_frames, \
        text_lengths, mels, max_len, output_lengths, embeds = inputs

        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        embeds = torch.unsqueeze(embeds, 1)
        embeds = embeds.expand(-1, encoder_outputs.size(1), -1)

        # Concatenating encoder_outputs with speaker embeddings
        encoder_outputs = torch.cat([encoder_outputs, embeds], dim=-1)

        # Predicting phoneme durations
        durations_pred = self.duration_predictor(encoder_outputs, text_lengths)
        durations_pred = durations_pred.squeeze(-1)

        # Predicting range parameters
        range_durations = durations_in_frames.unsqueeze(2) / 20
        range_predictor_input = torch.cat((range_durations, encoder_outputs), -1)

        range_pred = self.range_predictor(range_predictor_input, text_lengths)

        # Calling decoder to get mel spectrogram
        mel_outputs, alignments = self.decoder(
            encoder_outputs, durations_in_frames, duration_frames_indices,
            range_pred, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, durations_pred, alignments],
            output_lengths, text_lengths)

    def inference(self, inputs: torch.Tensor, embed: torch.Tensor) -> list:
        """
        Performs model forward in inference mode
        :param inputs: ids of input phonemes
        :param embed: speaker embedding vector
        :return: list of padded mel, post-net mel, alignment and duration
        """
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        # Concatenating embeddings with encoder outputs
        embeds = torch.unsqueeze(embed, 1)
        embeds = embeds.expand(-1, encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat([encoder_outputs, embeds], dim=-1)

        predicted_durations = self.duration_predictor.inference(encoder_outputs)

        # Bringing duration to original scale
        predicted_durations = predicted_durations * self.durations_length_std + self.durations_length_mean
        previous_step = 0
        durations_in_frames = []
        unpacked_durations = []

        for duration in predicted_durations[0, :, 0]:
            # Converts durations in second to durations in frames
            spec_frame_true = duration * self.sampling_rate / self.hop_length
            spec_frame_true = spec_frame_true + previous_step
            spec_frame = int(spec_frame_true)
            previous_step = spec_frame_true - spec_frame
            durations_in_frames.append(spec_frame)
        durations_in_frames[-1] += 1

        for duration in durations_in_frames:
            unpacked_durations.append(torch.arange(duration) + 1)

        unpacked_durations = torch.cat(unpacked_durations, dim=-1)
        durations_in_frames = torch.LongTensor(durations_in_frames)

        unpacked_durations = unpacked_durations.unsqueeze(0)
        durations_in_frames = durations_in_frames.unsqueeze(0)

        # Predicting range parameters
        range_durations = durations_in_frames.unsqueeze(2) / 20
        range_predictor_input = torch.cat((range_durations, encoder_outputs), -1)
        range_pred = self.range_predictor.inference(range_predictor_input)

        mel_outputs, alignments = self.decoder.inference(encoder_outputs,
                                                         durations_in_frames,
                                                         unpacked_durations,
                                                         range_pred)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, predicted_durations, alignments])

        return outputs

    def get_duration_stats(self):
        with open(os.path.join(self.dataset_path, 'duration_stats.json'), 'r') as f:
            duration_stats = json.load(f)
        return duration_stats["durations_mean"], duration_stats["durations_std"]