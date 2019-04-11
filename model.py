import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderNet(nn.Module):
    def __init__(self, embed_size):
        super(EncoderNet, self).__init__()
        # self.n_class = N_CLASS
        self.embed_size = embed_size

        self.encoder_fc_1_3 = nn.Sequential(
            #########################################
            ###        TODO: Add more layers      ###
            #########################################

            # fc1 1/2 = 256/2 = 128 => N * 64 * 128 * 128
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # fc2 1/4 = 128/2 = 64 => N * 128 * 64 * 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # fc3 1/8 = 64/2 = 32 => N * 256 * 32 * 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder_fc_4 = nn.Sequential(
            # fc4 1/16 = 32/2 = 16 => N * 512 * 16 * 16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder_fc_5 = nn.Sequential(
            # fc5 1/32 = 16/2 = 8 => N * 512 * 8 * 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.encoder_fc_6_7 = nn.Sequential(
            # fc6 N * 4096
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7 1 * 1 conv => N * 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.fc = nn.Linear(4096, self.embed_size)

        # fc8 skip connection
        # self.score_fc7 = nn.Conv2d(4096, self.n_class, 1)
        # self.score_pool3 = nn.Conv2d(256, self.n_class, 1)
        # self.score_pool4 = nn.Conv2d(512, self.n_class, 1)
        #
        # # fc9
        # self.fc9 = nn.ConvTranspose2d(self.n_class, self.n_class, kernel_size=4, stride=2, bias=False)
        #
        # # fc10
        # self.fc10 = nn.ConvTranspose2d(self.n_class, self.n_class, kernel_size=4, stride=2, bias=False)
        #
        # # fc11
        # self.fc11 = nn.ConvTranspose2d(self.n_class, self.n_class, kernel_size=16, stride=8, bias=False)

        self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.zero_()
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         if isinstance(m, nn.ConvTranspose2d):
    #             assert m.kernel_size[0] == m.kernel_size[1]
    #             initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
    #             m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self, vgg16):
        encode_list = [self.encoder_fc_1_3, self.encoder_fc_4, self.encoder_fc_5]
        features = [layer for seq in encode_list for layer in seq]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for idx in [0, 3]:
            l1 = vgg16.classifier[idx]
            l2 = getattr(self, 'encoder_fc_6_7')[idx]
            l2.weight.data = l1.weight.data  #.view(l2.weight.size())
            l2.bias.data = l1.bias.data  #.view(l2.bias.size())

    def forward(self, x):
        # x = self.layers(x)

        # layer3
        h = x
        h = self.encoder_fc_1_3(h)
        # pool3 = h

        # layer4
        h = self.encoder_fc_4(h)
        # pool4 = h

        # layer 5
        h = self.encoder_fc_5(h)

        h=h.view(h.size[0],-1)

        # encode finish
        h = self.encoder_fc_6_7(h)

        h = self.fc(h)

        # layer8, 9
        # h = self.fc9(self.score_fc7(h))
        # upsample_32 = h
        #
        # # layer8, 10
        # h = self.score_pool4(pool4)
        # h = h[:, :, 5:5 + upsample_32.size()[2], 5:5 + upsample_32.size()[3]]
        # score_pool4 = h
        #
        # h = upsample_32 + score_pool4
        # h = self.fc10(h)
        # upsample_16 = h
        #
        # # layer8, 11
        # h = self.score_pool3(pool3)
        # h = h[:, :, 9:9 + upsample_16.size()[2], 9:9 + upsample_16.size()[3]]
        # score_pool3 = h
        #
        # h = upsample_16 + score_pool3
        # h = self.fc11(h)

        # h = h[:, :, 28:28 + x.size()[2], 28:28 + x.size()[3]].contiguous()

        return h


class DecoderNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderNet, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(embeddings)  # packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def predict(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids




