import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter

from libs.models import network
from libs.utils import data_loader
from libs.utils.logger import Logger

params = {}

def test_net(params):
    # Create network
    slip_detection_model = network.Slip_detection_network(base_network=params['cnn'], pretrained=params['pretrained'],
                                                          rnn_input_size=params['rnn_input_size'],
                                                          rnn_hidden_size=params['rnn_hidden_size'],
                                                          rnn_num_layers=params['num_layers'],
                                                          num_classes=params['num_classes'],
                                                          use_gpu=params['use_gpu'],
                                                          dropout=params['dropout'])
    if params['use_gpu']:
        slip_detection_model = slip_detection_model.cuda()

    assert 'net_params' in params.keys(), "Must set network dir."
    assert params['net_params'].endswith('.pth'), "Wrong model path {}".format(params['net_params'])
    net_params_state_dict = torch.load(params['net_params'])
    slip_detection_model.load_state_dict(net_params_state_dict)

    test_dataset = data_loader.Tacile_Vision_dataset(data_path=params['test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                                  num_workers=params['num_workers'])

    with torch.no_grad():
        correct = 0
        total = 0
        for rgb_imgs, tacitle_imgs, labels in test_data_loader:
            outputs = slip_detection_model(rgb_imgs, tacitle_imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))


if __name__ == '__main__':
    # No modification is recommended.
    params['rnn_input_size'] = 64
    params['rnn_hidden_size'] = 64
    params['num_classes'] = 2
    params['num_layers'] = 1
    params['pretrained'] = False # CNN is pretrained by ImageNet or not
    params['batch_size'] = 1

    # Customer params setting.
    params['epochs'] = 10
    params['print_interval'] = 5
    params['num_workers'] = 1
    params['use_gpu'] = False
    params['lr'] = 1e-5
    params['dropout'] = 0.8
    params['test_data_dir'] = 'data'
    # Use Alextnet to debug.
    # You can choose vgg_16, vgg_19 or inception_v3(unreliable). Poor MBP
    params['cnn'] = 'debug'
    params['net_params'] = 'model/slip_detection_network_00009.pth'

    # params['save_dir'] = 'model'
    # Start train
    test_net(params)

