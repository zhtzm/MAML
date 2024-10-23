import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import import_class_from_string, Averager, plot_training_curves, parse_yaml_config


def fine(config):
    dataset_class, _ = import_class_from_string(config['dataset']['target'])
    dataset = dataset_class(**config['dataset']['params'])
    train_dataset, val_dataset = dataset.split(0.6, 0.4)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=8)

    model_cfg = config['model']
    model_class, _ = import_class_from_string(model_cfg['target'])
    model = model_class(**model_cfg['params'])
    if model_cfg['init_weight'] is not None:
        model.load_pretrained_weights(model_cfg['init_weight'])

    optimizer_class, _ = import_class_from_string(config['optimizer']['target'])
    optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])
    if config['scheduler'] is not None:
        scheduler_class, _ = import_class_from_string(config['scheduler']['target'])
        scheduler = scheduler_class(optimizer, **config['scheduler']['params'])
    else:
        scheduler = None

    model = model.cuda()

    trlog = {'train_loss': [],
             'train_acc': [],
             'val_loss': [],
             'val_acc': []}

    for epoch in range(config['max_epoch']):
        model.train()
        tl = Averager()
        ta = Averager()
        vl = Averager()
        va = Averager()

        train_bar = tqdm(train_loader)
        for features, labels in train_bar:
            train_bar.set_description("epoch {}".format(epoch + 1))

            features = features.cuda()
            labels = labels.cuda()

            logit = model(features)
            prediction = torch.argmax(logit, dim=1)
            loss = nn.CrossEntropyLoss().cuda()(logit, labels)
            acc = torch.eq(labels, prediction).sum() / len(labels)

            tl.add(loss.item())
            ta.add(acc.item())
            train_bar.set_postfix(loss="{:.4f}".format(tl.item()), acc="{:.4f}".format(ta.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for features, labels in val_bar:
                val_bar.set_description("epoch {}".format(epoch + 1))

                features = features.cuda()
                labels = labels.cuda()

                logit = model(features)
                prediction = torch.argmax(logit, dim=1)
                loss = nn.CrossEntropyLoss().cuda()(logit, labels)
                acc = torch.eq(labels, prediction).sum() / len(labels)

                vl.add(loss.item())
                va.add(acc.item())
                val_bar.set_postfix(loss="{:.4f}".format(vl.item()), acc="{:.4f}".format(va.item()))

        trlog['train_loss'].append(tl.item())
        trlog['train_acc'].append(ta.item())
        trlog['val_loss'].append(vl.item())
        trlog['val_acc'].append(va.item())

        torch.save(trlog, os.path.join('./results', 'trlog-' + config['exp']))

    plot_training_curves(trlog)


if __name__ == '__main__':
    cfg = parse_yaml_config('yaml/FTD_90_200_fMRI.yaml')
    fine(cfg)