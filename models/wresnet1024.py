import logging
import torch
import torch.nn as nn
from models.wider_resnet import wresnet
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

logger = logging.getLogger(__name__)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size[0] // 2,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(cur_layer_input.size(1)):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not isinstance(kernel_size, tuple):
            raise ValueError('`kernel_size` must be a tuple')


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info('=> ' + self.model_name + '_1024: (CATTN + ConvLSTM) - Ped2')

        self.wrn38 = wresnet(config, self.model_name, pretrained=pretrained)  # wrn38(config, pretrained=True)

        channels = [4096, 2048, 1024, 512, 256, 128]

        self.conv_x8 = nn.Conv2d(channels[0] * frames, channels[1], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[4] * frames, channels[4], kernel_size=1, bias=False)
        self.conv_x1 = nn.Conv2d(channels[5] * frames, channels[5], kernel_size=1, bias=False)

        self.up8 = ConvTransposeBnRelu(channels[1], channels[2], kernel_size=2)   # 2048          -> 1024
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[4], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[5], channels[4], kernel_size=2)   # 512   +   128 -> 256

        # Replacing TSM with ConvLSTM
        self.conv_lstm = ConvLSTM(input_dim=2048, hidden_dim=2048, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.attn8 = ChannelAttention(channels[2])
        self.attn4 = ChannelAttention(channels[3])
        self.attn2 = ChannelAttention(channels[4])

        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=1, padding=0),
            ConvBnRelu(channels[5], channels[5], kernel_size=3, padding=1),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        x1s, x2s, x8s = [], [], []
        for xi in x:
            x1, x2, x8 = self.wrn38(xi)
            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)

        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))

        # ConvLSTM replacing TSM
        lstm_out, _ = self.conv_lstm(x8.unsqueeze(1))  # Add sequence dimension
        x8 = lstm_out[0].squeeze(1)  # Extract first layer output and squeeze sequence dimension

        x = self.up8(x8)
        x = self.attn8(x)

        x = self.up4(torch.cat([x2, x], dim=1))
        x = self.attn4(x)

        x = self.up2(torch.cat([x1, x], dim=1))
        x = self.attn2(x)

        return self.final(x)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//reduction, input_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y
