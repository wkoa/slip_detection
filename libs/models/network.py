import torch
from torch import nn
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Basic_network(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False,):
        super(Basic_network, self).__init__()
        # Define CNN to extract features.
        self.features = None
        if base_network == 'vgg_16':
            self.features = vgg16_bn(pretrained=pretrained)
            # To delete fc8
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            self.fc = nn.Sequential(nn.Linear(4096*2, 64))
        elif base_network == 'vgg_19':
            self.features = vgg19_bn(pretrained=pretrained)
            # To delete fc8
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            self.fc = nn.Sequential(nn.Linear(4096*2, 64))
        elif base_network == 'inception_v3':
            #TODO It is unreliable.
            self.features = inception_v3(pretrained=pretrained)
            # To delete the last layer.
            self.features.fc = nn.Sequential(*list(self.features.fc.children())[:-1])
            self.fc = nn.Sequential(nn.Linear(2048*2, 64))
        elif base_network == 'debug':
            self.features = alexnet(pretrained=pretrained)
            # To delete the last layer
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            self.fc = nn.Sequential(nn.Linear(4096*2, 64))

        assert self.features, "Illegal CNN network name!"

    def forward(self, x_1, x_2):
        features_1 = self.features(x_1)
        features_2 = self.features(x_2)

        features = torch.cat((features_1, features_2), 1)
        features = features.view(features.size(0), -1)
        features = self.fc(features)

        return features


# RNN Classifier
class RNN_network(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=1, num_classes=2, use_gpu=False, dropout=0.8):
        super(RNN_network, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # Set initial hidden and cell states
        if self.use_gpu:
            h0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, len(x), self.hidden_size)
            c0 = torch.zeros(self.num_layers, len(x), self.hidden_size)

        # Forward propagate LSTM
        x, _ = self.lstm_1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.dropout_1(x)
        out, _ = self.lstm_2(x, (h0, c0))
        out = self.dropout_2(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out


class Slip_detection_network(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False, rnn_input_size=64, rnn_hidden_size=64,
                 rnn_num_layers=1, num_classes=2, use_gpu=False, dropout=0.8):
        super(Slip_detection_network, self).__init__()
        self.cnn_network = Basic_network(base_network=base_network,pretrained=pretrained)
        self.rnn_network = RNN_network(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                       num_classes=num_classes, use_gpu=use_gpu, dropout=dropout)
        self.use_gpu = use_gpu

    def forward(self, x_1, x_2):
        """
        :param x_1: a list of 8 rgb imgs(tensor)
        :param x_2: a list of 8 tactile imgs(tensor)
        :return: network output
        """
        cnn_features = []
        for i in range(8):
            if self.use_gpu:
                features = self.cnn_network(x_1[i].to(device), x_2[i].to(device))
            else:
                features = self.cnn_network(x_1[i], x_2[i])
            cnn_features.append(features.tolist())
        cnn_features = torch.FloatTensor(cnn_features)
        if self.use_gpu:
            cnn_features = cnn_features.to(device)
        cnn_features = cnn_features.reshape([-1, 8, 64])
        output = self.rnn_network(cnn_features)

        return output


if __name__ == "__main__":
    network = Slip_detection_network()
    print(network)