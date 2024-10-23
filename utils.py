import importlib
import os
import pprint

import numpy as np
import torch
from matplotlib import pyplot as plt

import yaml

_utils_pp = pprint.PrettyPrinter()

class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def save_model(model, params, name, save_path):
    torch.save(dict(params=params, state_dict=model.state_dict()),
               str(os.path.join(save_path, name + '.pth')))


def plot_training_curves(trlog):
    epochs = range(1, len(trlog['train_loss']) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, trlog['train_loss'], label='Training Loss')
    plt.plot(epochs, trlog['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    if 'train_acc' in trlog and 'val_acc' in trlog:
        # 绘制准确率
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, trlog['train_acc'], label='Training Accuracy')
        plt.plot(epochs, trlog['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()


def import_class_from_string(class_string: str):
    module_name, class_name = class_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name), class_name
    except ImportError as e:
        raise ImportError(f"Could not import {class_string}. Reason: {e}")
    except AttributeError:
        raise AttributeError(f"Class {class_name} not found in module {module_name}.")


def parse_yaml_config(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise IOError(f"Configuration file {config_path} does not exist")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")


def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def count_acc(logistic, label):
    pred = torch.argmax(logistic, dim=1)
    # print(pred.size())
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


if __name__ == '__main__':
    log = torch.load('results/trlog-FTD_90_200_fMRI')
    plot_training_curves(log)
    print(np.argmax(np.array(log['val_acc'])), np.max(np.array(log['val_acc'])), np.mean(log['val_acc'][-10:]))
