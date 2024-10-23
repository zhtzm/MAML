import collections
import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import Averager, save_model, plot_training_curves


class MAMLLearn:
    def __init__(self, learning_net, save_path, model_params, inner_lr, inner_step):
        super(MAMLLearn, self).__init__()
        self.learning_net = learning_net
        self.save_path = save_path
        self.params = model_params
        self.inner_lr = inner_lr
        self.inner_step = inner_step

    def train(self, epochs, train_loader, val_loader, optimizer, scheduler=None):
        self.learning_net = self.learning_net.cuda()

        trlog = {'train_loss': [],
                 'train_acc': [],
                 'val_loss': [],
                 'val_acc': [],
                 'best_acc': 0.,
                 'best_acc_epoch': 0}
        writer = SummaryWriter(comment=self.save_path)

        for epoch in range(epochs):
            self.learning_net.train()
            tl = Averager()
            ta = Averager()
            vl = Averager()
            va = Averager()

            train_bar = tqdm(train_loader)
            for support_images, support_labels, query_images, query_labels in train_bar:
                train_bar.set_description("epoch {}/{}".format(epoch + 1, epochs))

                support_images = support_images.cuda()
                support_labels = support_labels.cuda()
                query_images = query_images.cuda()
                query_labels = query_labels.cuda()

                loss, acc = self.step(support_images, support_labels, query_images, query_labels,
                                       self.inner_step, optimizer, scheduler)

                tl.add(loss.item())
                ta.add(acc.item())
                if scheduler is not None:
                    train_bar.set_postfix(loss="{:.4f}".format(tl.item()), acc="{:.4f}".format(ta.item()),
                                          lr="{:.4f}".format(scheduler.get_last_lr()[0]))
                else:
                    train_bar.set_postfix(loss="{:.4f}".format(tl.item()), acc="{:.4f}".format(ta.item()))

            val_bar = tqdm(val_loader)
            for support_images, support_labels, query_images, query_labels in val_bar:
                val_bar.set_description("epoch {}/{}".format(epoch + 1, epochs))

                support_images = support_images.cuda()
                support_labels = support_labels.cuda()
                query_images = query_images.cuda()
                query_labels = query_labels.cuda()

                loss, acc = self.step(support_images, support_labels, query_images, query_labels,
                                       self.inner_step, optimizer, scheduler, is_train=False)

                vl.add(loss.item())
                va.add(acc.item())
                val_bar.set_postfix(loss="{:.4f}".format(vl.item()), acc="{:.4f}".format(va.item()))

            if va.item() > trlog['best_acc']:
                trlog['best_acc'] = va.item()
                trlog['best_acc_epoch'] = epoch
                save_model(self.learning_net, self.params, 'best_acc', self.save_path)

            trlog['train_loss'].append(tl.item())
            trlog['train_acc'].append(ta.item())
            trlog['val_loss'].append(vl.item())
            trlog['val_acc'].append(va.item())

            torch.save(trlog, os.path.join(self.save_path, 'trlog'))
            save_model(self.learning_net, self.params, 'epoch-last', self.save_path)

        writer.close()
        plot_training_curves(trlog)

    def step(self, support_images, support_labels, query_images, query_labels,
             inner_step, optimizer, scheduler, is_train=True):
        meta_loss = []
        meta_acc = []

        for support_image, support_label, query_image, query_label \
                in zip(support_images, support_labels, query_images, query_labels):
            fast_weights = collections.OrderedDict(self.learning_net.named_parameters())
            for _ in range(inner_step):
                support_logit = self.learning_net.functional_forward(support_image, fast_weights)
                support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)

                grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
                fast_weights = collections.OrderedDict((name, param - self.inner_lr * grad)
                                                       for ((name, param), grad) in zip(fast_weights.items(), grads))

            query_logit = self.learning_net.functional_forward(query_image, fast_weights)
            query_prediction = torch.argmax(query_logit, dim=1)

            query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
            query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

            meta_loss.append(query_loss)
            meta_acc.append(query_acc.data.cpu().numpy())

        meta_loss = torch.stack(meta_loss).mean()
        meta_acc = np.mean(meta_acc)

        if is_train:
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        return meta_loss, meta_acc
