import os

import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter

from libs.models import network
from libs.utils import data_loader

params = {}


def train_net(params):

    writer = SummaryWriter(log_dir='logs')
    dummy_input = [torch.zeros(1,3,480,640) for i in range(8)]
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
    # Some Warnings in there.
    # writer.add_graph(slip_detection_model, input_to_model=(dummy_input, dummy_input))

    if 'net_params' in params.keys():
        assert params['net_params'].endswith('.pth'), "Wrong model path {}".format(params['net_params'])
        net_params_state_dict = torch.load(params['net_params'])
        slip_detection_model.load_state_dict(net_params_state_dict)

    # Init optimizer & loss func.
    optimizer = optim.Adam(slip_detection_model.rnn_network.parameters(), lr=params['lr'])
    loss_function = nn.CrossEntropyLoss()

    # Dataloader
    train_dataset = data_loader.Tacile_Vision_dataset(data_path=params['train_data_dir'])
    train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                   num_workers=params['num_workers'])
    test_dataset = data_loader.Tacile_Vision_dataset(data_path=params['test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=params['num_workers'])
    # To record training procession
    train_loss = []
    train_acc = []

    # Start training
    for epoch in range(params['epochs']):
        # Start
        total_loss = 0.0
        total_acc = 0.0
        total = 0.0
        for i, data in enumerate(train_data_loader):
            # one iteration
            rgb_imgs, tacitle_imgs, label = data
            output = slip_detection_model(rgb_imgs, tacitle_imgs)
            loss = loss_function(output, label)

            # Backward & optimize
            slip_detection_model.zero_grad()
            loss.backward()
            optimizer.step()

            # cal training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == label).sum().item()
            total_loss += float(loss.data)
            total += len(label)
        train_loss.append(total_loss/total)
        train_acc.append(total_acc/total)

        writer.add_scalar('train_loss', train_loss[epoch],)
        writer.add_scalar('train_acc', train_acc[epoch],)
        if epoch%params['print_interval'] == 0:
            print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f'
                  % (epoch, params['epochs'], train_loss[epoch], train_acc[epoch],))
        if (epoch + 1)%params['test_interval'] == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for rgb_imgs, tacitle_imgs, labels in test_data_loader:
                    outputs = slip_detection_model(rgb_imgs, tacitle_imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        # Save 5 different model
        if epoch%(int(params['epochs']/5)) == 0:
            if 'save_dir' in params.keys():
                model_path = os.path.join(params['save_dir'], 'slip_detection_network_{:0>5}.pth'.format(epoch))
                torch.save(slip_detection_model.state_dict(), model_path)

    if 'save_dir' in params.keys():
        model_path = os.path.join(params['save_dir'], 'slip_detection_network_{:0>6}.pth'.format(epoch))
        torch.save(slip_detection_model.state_dict(), model_path)
    writer.close()


if __name__ == '__main__':
    # No modification is recommended.
    params['rnn_input_size'] = 64
    params['rnn_hidden_size'] = 64
    params['num_classes'] = 2
    params['num_layers'] = 1

    # Customer params setting.
    params['epochs'] = 10
    params['print_interval'] = 5
    params['test_interval'] = 10
    params['batch_size'] = 2
    params['num_workers'] = 1
    params['use_gpu'] = False
    params['lr'] = 1e-5
    params['dropout'] = 0.8
    params['train_data_dir'] = 'data'
    params['test_data_dir'] = 'data'
    # Use Alextnet to debug.
    # You can choose vgg_16, vgg_19 or inception_v3(unreliable). Poor MBP
    params['cnn'] = 'debug'
    params['pretrained'] = False # CNN is pretrained by ImageNet or not
    # params['net_params'] = 'model/pretrained_net/'

    params['save_dir'] = 'model'
    # Start train
    train_net(params)


