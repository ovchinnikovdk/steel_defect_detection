import argparse
from torch.utils.data import DataLoader
from lib.configs import ConfigFactory
from lib.mask_utils import pred2mask
import os

import torch
import numpy as np
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_conf", type=str, help='JSON File Model Configuration')
    parser.add_argument("--train_conf", type=str, help='JSON File Train configuration')

    args = parser.parse_args()
    configurator = ConfigFactory()
    model = configurator.build_model(args.model_conf)
    train_configs = configurator.build_train_env(model, args.train_conf)
    train(**train_configs)


def train(net, loss, metrics, train_data, valid_data, optimizer, gpu, batch_size=64, epochs=30, log_path='logs'):
    val_loader = DataLoader(valid_data, batch_size=batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    net.train()
    score_history = [0.]
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
        print("Loss: " + str(sum_loss))
        validate(net, val_loader, metrics, loss, score_history, scheduler, gpu, log_path, i)


def validate(net, val_loader, metrics, loss, score_history, scheduler, gpu, log_path, epoch):
    # Validating Epoch
    torch.cuda.empty_cache()
    net.eval()
    pred_y = []
    true_y = []
    val_score = dict()
    with torch.no_grad():
        val_loss = 0.
        for val_x, val_y in tqdm.tqdm(val_loader, desc='Validating epoch #' + str(epoch)):
            if gpu:
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            pred = net(val_x)
            loss_out = loss(pred, val_y)
            val_loss += loss_out.item()
            pred_y.append(pred2mask(pred.cpu()))
            true_y.append(val_y.cpu())
            torch.cuda.empty_cache()
        pred_y = torch.cat(pred_y, dim=0)
        true_y = torch.cat(true_y, dim=0)
        print("Validation loss: {0:10.5f}".format(val_loss))
        print(pred_y.shape)
        for metric in metrics.keys():
            val_score[metric] = metrics[metric](pred_y, true_y)
        print(val_score)
        val_score = np.mean(list(val_score.values()))
        if val_score > max(score_history):
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            torch.save(net.state_dict(), os.path.join(log_path, net.__class__.__name__ + '_best.dat'))
        score_history.append(val_score)
        scheduler.step(val_score)


if __name__ == '__main__':
    main()
