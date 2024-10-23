import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet:
    def __init__(self,
                 data_path,
                 num_tasks,
                 n_way,
                 k_shot,
                 q_query):
        self.data_path = data_path
        self.num_tasks = num_tasks
        self.train_path = osp.join(data_path, 'train')
        self.val_path = osp.join(data_path, 'val')
        self.test_path = osp.join(data_path, 'test')
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        ])

    def make_dataset(self, path):
        data = []
        labels = []
        categories_map = {}
        idx = 0

        for class_name in os.listdir(path):
            class_path = osp.join(path, class_name)
            if not os.path.isdir(class_path):
                continue

            categories_map[idx] = class_name
            for img_name in os.listdir(class_path):
                img_path = osp.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append(img_path)
                    labels.append(idx)
            idx += 1

        dataset = MiniImageNetMetaDataset(data, labels, categories_map,
                                          self.num_tasks,
                                          self.transform,
                                          self.n_way, self.k_shot, self.q_query)

        return dataset

    def get_train_dataset(self):
        return self.make_dataset(self.train_path)

    def get_val_dataset(self):
        return self.make_dataset(self.val_path)

    def get_test_dataset(self):
        return self.make_dataset(self.test_path)


class MiniImageNetMetaDataset(Dataset):
    def __init__(self, data, labels, categories_map, batch_size, transform,
                 n_way, k_shot, q_query,):
        self.data = data
        self.labels = labels
        self.categories_map = categories_map
        self.batch_size = batch_size
        self.num_classes = len(categories_map)
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        labels = np.array(labels)
        self.class_indices = {i: np.argwhere(labels == i).reshape(-1) for i in range(self.num_classes)}

    def __len__(self):
        return self.batch_size * 32

    def __getitem__(self, idx):
        def create_data_samples(classes, k_shot, q_query):
            support = []
            query = []
            for label, class_id in enumerate(classes):
                class_indices = self.class_indices[class_id]
                samples = random.sample(class_indices.tolist(), k_shot + q_query)
                for index in samples[:k_shot]:
                    image = self.transform(Image.open(self.data[index]).convert('RGB'))
                    image_feature = image.reshape(-1)
                    support.append((image_feature, label))
                for index in samples[k_shot:]:
                    image = self.transform(Image.open(self.data[index]).convert('RGB'))
                    image_feature = image.reshape(-1)
                    query.append((image_feature, label))
            return support, query

        def separate_images_and_labels(data):
            images = [item[0] for item in data]
            labels = [item[1] for item in data]
            return torch.stack(images), torch.tensor(labels, dtype=torch.long)

        selected_classes = random.sample(range(self.num_classes), self.n_way)
        support_data, query_data = create_data_samples(selected_classes, self.k_shot, self.q_query)

        random.shuffle(support_data)
        random.shuffle(query_data)

        support_images, support_labels = separate_images_and_labels(support_data)
        query_images, query_labels = separate_images_and_labels(query_data)

        return support_images, support_labels, query_images, query_labels
