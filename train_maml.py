import os.path as osp

from torch.utils.data import DataLoader

from datasets_class.mat_dataset import BaseMatDataset
from datasets_class.mini_imagenet import MiniImageNet
from maml import MAMLLearn
from utils import ensure_path, parse_yaml_config, import_class_from_string


def train_siamese_meta(config):
    save_path = osp.join('./results', config.get("exp"))
    ensure_path(save_path)

    dataset_class, _ = import_class_from_string(config['dataset']['target'])
    dataset = dataset_class(**config['dataset']['params'])
    if isinstance(dataset, MiniImageNet):
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_val_dataset()
    elif isinstance(dataset, BaseMatDataset):
        train_dataset, val_dataset = dataset.split_for_maml(0.6, 0.4)
    else:
        raise NotImplementedError

    batch_size = config['dataset']['params']['num_tasks']
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8)

    model_cfg = config['model']
    model_class, _ = import_class_from_string(model_cfg['target'])
    model = model_class(**model_cfg['params'])

    if model_cfg['init_weight'] is not None:
        model.load_pretrained_weights(model_cfg['init_weight'])

    optimizer_class, _ = import_class_from_string(config['outer_optimizer']['target'])
    optimizer = optimizer_class(model.parameters(), **config['outer_optimizer']['params'])
    if config['scheduler'] is not None:
        scheduler_class, _ = import_class_from_string(config['scheduler']['target'])
        scheduler = scheduler_class(optimizer, **config['scheduler']['params'])
    else:
        scheduler = None

    maml = MAMLLearn(model, save_path, model_cfg['params'], config['inner_lr'], config['inner_step'])
    maml.train(config['max_epoch'], train_loader, val_loader, optimizer, scheduler)


if __name__ == '__main__':
    cfg = parse_yaml_config('yaml/mini_imagenet-pretrain-2way-1shot.yaml')
    train_siamese_meta(cfg)

    # cfg = parse_yaml_config('yaml/MAML_ADNI_90_120_fMRI.yaml')
    # train_siamese_meta(cfg)

    cfg = parse_yaml_config('yaml/MAML_FTD_90_200_fMRI.yaml')
    train_siamese_meta(cfg)
    cfg = parse_yaml_config('yaml/MAML_OCD_90_200_fMRI.yaml')
    train_siamese_meta(cfg)
    cfg = parse_yaml_config('yaml/MAML_PPMI.yaml')
    train_siamese_meta(cfg)
