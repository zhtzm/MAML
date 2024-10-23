import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import import_class_from_string


class Classifier(nn.Module):
    def __init__(self, input_dim=3 * 84 * 84, hidden_dim=512, output_dim=64):
        super(Classifier, self).__init__()
        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.block4 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.softmax(self.output(x), dim=1)
        return x

    def functional_forward(self, x, params):
        x = F.linear(x, params['input.weight'], params['input.bias'])
        x = F.relu(x)

        for i in range(4):
            x = F.linear(x, params[f'block{i + 1}.0.weight'], params[f'block{i + 1}.0.bias'])
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = F.linear(x, params['output.weight'], params['output.bias'])
        x = F.softmax(x, dim=1)
        return x

    def load_pretrained_weights(self, init_weight):
        model_dict = self.state_dict()
        if init_weight is not None:
            pretrained_dict = torch.load(init_weight, map_location='cpu')['state_dict']

            # 先过滤出其它层的权重
            filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                        k in model_dict and k not in
                                        ['input.weight', 'input.bias']}

            model_dict.update(filtered_pretrained_dict)

            if self.input.in_features == pretrained_dict['input.weight'].size(1):
                model_dict['input.weight'] = pretrained_dict['input.weight']
                model_dict['input.bias'] = pretrained_dict['input.bias']

            self.load_state_dict(model_dict)

    def create_optimizers(self, optimizer_cfg):
        sp_params = list(self.input.parameters())
        optimizer_class, _ = import_class_from_string(optimizer_cfg['optimizer_sp']['target'])
        optimizer_sp = optim.Adam(sp_params, **optimizer_cfg['optimizer_sp']['params'])

        other_params = [p for p in self.parameters() if id(p) not in [id(sp) for sp in sp_params]]
        optimizer_class, _ = import_class_from_string(optimizer_cfg['optimizer_other']['target'])
        optimizer_other = optim.Adam(other_params, **optimizer_cfg['optimizer_other']['params'])

        return optimizer_sp, optimizer_other
