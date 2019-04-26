import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderNet(nn.Module):
    def __init__(self, embed_size):
        super(EncoderNet, self).__init__()
        # self.n_class = N_CLASS
        self.embed_size = embed_size

        ############### VGG16 version #################
        # self.encoder_fc_1_3 = nn.Sequential(
        #     #########################################
        #     ###        TODO: Add more layers      ###
        #     #########################################
        #
        #     # fc1 1/2 = 256/2 = 128 => N * 64 * 128 * 128
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     # fc2 1/4 = 128/2 = 64 => N * 128 * 64 * 64
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     # fc3 1/8 = 64/2 = 32 => N * 256 * 32 * 32
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.encoder_fc_4 = nn.Sequential(
        #     # fc4 1/16 = 32/2 = 16 => N * 512 * 16 * 16
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.encoder_fc_5 = nn.Sequential(
        #     # fc5 1/32 = 16/2 = 8 => N * 512 * 8 * 8
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        #
        # self.encoder_fc_6_7 = nn.Sequential(
        #     # fc6 N * 4096
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(),
        #
        #     # fc7 1 * 1 conv => N * 4096
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d()
        # )
        # self.fc = nn.Linear(4096, self.embed_size)

        ############### ResNet version #################
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

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
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

    def forward(self, x):
        ############### VGG16 version #################
        x = x.permute(0, 3, 1, 2)
        # h = x
        # h = self.encoder_fc_1_3(h)
        # h = self.encoder_fc_4(h)
        # h = self.encoder_fc_5(h)
        # h = h.view((h.shape[0], -1))
        # h = self.encoder_fc_6_7(h)

        ############### ResNet version #################
        with torch.no_grad():
            h = self.resnet(x)
        h = h.reshape(h.size(0), -1)
        # h = self.linear(h)
        h = self.fc(h)
        
        return h


class DecoderNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,  embeddic,max_seq_length=20):
        super(DecoderNet, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embeddic),freeze=False)
        # self.embed.weight.data.(torch.from_numpy(embeddic))
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        # print('embed1', embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # print('embed2', embeddings.shape)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)  # packed)
        # print(hiddens.shape)
        outputs = self.linear(hiddens[0])
        # outputs = outputs.permute(0, 2, 1)
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




