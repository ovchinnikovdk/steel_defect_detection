import argparse
from torch.utils.data import DataLoader
from lib.configs import ConfigFactory
from lib.mask_utils import pred2mask, save_mask_image
import os
import json
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


def train(net, loss, metrics, train_data,
          valid_data, optimizer, gpu,
          batch_size=64, epochs=30, log_path='logs',
          net_version='dummy1', show_predictions=False):
    val_loader = DataLoader(valid_data, batch_size=12, num_workers=4)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    net.train()
    score_history = [0.]
    val_loss_history = []
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
        validate(net,
                 val_loader,
                 metrics,
                 loss,
                 score_history,
                 val_loss_history,
                 scheduler,
                 gpu,
                 log_path,
                 i,
                 net_version,
                 show_predictions)
    torch.save(net.state_dict(), os.path.join(log_path, net_version + '_last.dat'))
    with open(os.path.join(log_path, 'params_last.json'), 'w') as params_file:
        json.dump({'lr': optimizer.state_dict()}, params_file)


def validate(net, val_loader, metrics, loss, score_history, loss_history, scheduler, gpu, log_path, epoch, net_version, show=False):
    # Validating Epoch
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    torch.cuda.empty_cache()
    net.eval()
    val_score = dict()
    with torch.no_grad():
        val_loss = 0.
        for val_x, val_y in tqdm.tqdm(val_loader, desc='Validating epoch #' + str(epoch)):
            if gpu:
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            pred = net(val_x)
            if isinstance(loss, torch.nn.BCEWithLogitsLoss) or hasattr(net, 'predict'):
                pred = torch.nn.Sigmoid()(pred)
            loss_out = loss(pred, val_y)
            val_loss += loss_out.item()
            pred = pred2mask(pred.cpu(), 0.5)
            if show:
                save_mask_image(log_path, val_x, val_y, pred)
            for metric in metrics.keys():
                if metric in val_score:
                    val_score[metric].append(metrics[metric](pred, val_y.cpu()))
                else:
                    val_score[metric] = [metrics[metric](pred, val_y.cpu())]
            torch.cuda.empty_cache()
        print("Validation loss: {0:10.5f}".format(val_loss))
        for metric in metrics.keys():
            val_score[metric] = np.mean(val_score[metric]) # metrics[metric](pred_y, true_y)
        print(val_score)
        val_score_mean = np.mean(list(val_score.values()))
        if val_score_mean > max(score_history) or (val_score_mean == max(score_history) and val_loss < min(loss_history)):
            torch.save(net.state_dict(), os.path.join(log_path, net_version + '_best.dat'))
            with open(os.path.join(log_path, 'params_best.json'), 'w') as params_file:
                json.dump({'epoch': epoch, 'lr': scheduler.state_dict(), 'metrics': val_score}, params_file)
        score_history.append(val_score_mean)
        loss_history.append(val_loss)
        with open(os.path.join(log_path, 'score_history.json'), 'w') as history_file:
            json.dump(score_history, history_file)
        with open(os.path.join(log_path, 'loss_history.json'), 'w') as history_file:
            json.dump(loss_history, history_file)

        scheduler.step(val_score_mean)


if __name__ == '__main__':
    main()
