import argparse
from torch.utils.data import DataLoader
from lib.configs import ConfigFactory
import os

import torch
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_conf", type=str, help='JSON File Model Configuration')
    parser.add_argument("--train_conf", type=str, help='JSON File Train configuration')
    # parser.add_argument("log_path")

    args = parser.parse_args()
    configurator = ConfigFactory()
    model = configurator.build_model(args.model_conf)
    train_configs = configurator.build_train_env(model, args.train_conf)
    train(**train_configs)


def train(net, loss, metrics, train_data, valid_data, optimizer, gpu, batch_size=64, epochs=30, log_path='/logs'):
    val_loader = DataLoader(valid_data, batch_size=batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    net.train()
    for i in range(epochs):
        print("EPOCH #" + str(i) + ' of ' + str(epochs))
        net.train()
        sum_loss = 0
        for x, y in tqdm.tqdm(train_loader, desc='Training epoch #' + str(i)):
            if gpu:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            output = net(x)
            loss_out = loss(output, y)
            loss_out.backward()
            optimizer.step()
            sum_loss += loss_out.item()
        # Validating Epoch
        print("Loss: " + str(sum_loss))
        net.eval()
        pred_y = []
        true_y = []
        val_loss = 0
        for val_x, val_y in tqdm.tqdm(val_loader, desc='Validating epoch #' + str(i)):
            if gpu:
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            pred = net(val_x)
            loss_out = loss(pred, val_y)
            val_loss += loss_out.item()
            pred_y.append(pred.cpu().detach().numpy()[0])
            true_y.append(val_y)
        val_score = dict()
        for metric in metrics.keys():
            val_score[metric] = metrics[metric](pred_y, true_y)
        print(val_score)
        torch.save(net.state_dict(), os.path.join(log_path, net.__class__.__name__ + str(i) + '.dat'))


if __name__ == '__main__':
    main()
