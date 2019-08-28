import os
import tqdm
import torch
from lib.mask_utils import pred2mask, save_mask_image
import numpy as np
import json


class TrainRunner(object):
    def __init__(self, net, train_loader, val_loader, loss, lr, metrics, log_path, net_name, use_gpu=True, show=False):
        self.net = net
        self.phases = ['train', 'validation']
        self.loaders = {'train': train_loader, 'validation': val_loader}
        self.loss = loss
        self.acc_steps = 32 // self.loaders['train'].batch_size
        self.acc_steps = self.acc_steps if self.acc_steps != 0 else 1
        self.lr = lr
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.metrics = metrics
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.net_name = net_name
        self.gpu = use_gpu
        self.show = show
        self.current_epoch = 0
        self.score_history = [0.]

    def iterate(self, phase):
        if phase == 'train':
            self.net.train()
        else:
            self.net.eval()
        sum_loss = 0
        self.optimizer.zero_grad()
        for itr, (x, y) in enumerate(tqdm.tqdm(self.loaders[phase], desc=f"Epoch #{self.current_epoch}, Phase: {phase}")):
            with torch.set_grad_enabled(phase == 'train'):
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                output = self.net(x)
                loss_out = self.loss(output, y)
                if phase == 'train':
                    loss_out.backward()
                    if (itr + 1) % self.acc_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    val_score = dict()
                    if isinstance(self.loss, torch.nn.BCEWithLogitsLoss) or hasattr(self.net, 'predict'):
                        pred = torch.nn.Sigmoid()(output)
                    pred = pred2mask(pred.cpu(), 0.5)
                    if self.show:
                        save_mask_image(self.log_path, x, y, pred)
                    for metric in self.metrics.keys():
                        if metric in val_score:
                            val_score[metric].append(self.metrics[metric](pred, y.cpu()))
                        else:
                            val_score[metric] = [self.metrics[metric](pred, y.cpu())]
                torch.cuda.empty_cache()
                sum_loss += loss_out.item()
        if phase != 'train':
            for metric in self.metrics.keys():
                val_score[metric] = np.mean(val_score[metric])
            print(val_score)
            score = np.mean(list((val_score.values())))
            self.score_history.append(score)
            if max(self.score_history) < score:
                print(f"Saving new best model, score: {score}")
                params = {'score': score, 'epoch': self.current_epoch}
                self.save_state(params, 'best')
        print(f"Loss ({phase}): " + str(sum_loss))
        return sum_loss

    def run(self, n_epochs):
        losses = {'train': [], 'validation': []}
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            for phase in self.phases:
                losses[phase].append(self.iterate(phase))
            self.scheduler.step(losses['validation'][-1], self.current_epoch)
        self.save_state({'scores': self.score_history, 'losses': losses}, 'last')

    def save_state(self, params, reason):
        torch.save(self.net.state_dict(), os.path.join(self.log_path, self.net_name + f"_{reason}.dat"))
        with open(os.path.join(self.log_path, f"params_{reason}.json"), 'w') as json_file:
            json.dump(params, json_file)

