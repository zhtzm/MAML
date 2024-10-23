import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class BaseMatDataset(Dataset):
    def __init__(self,
                 data_path,
                 categories,
                 num_tasks=0,
                 n_way=0,
                 k_shot=0,
                 q_query=0
                 ):
        self.data_path = data_path
        data, labels, self.categories_map = self._process_mat_file(categories)
        self.data = torch.from_numpy(data).float()
        self.data = self.data.reshape(data.shape[0], -1)
        self.labels = torch.from_numpy(labels).long()
        self.num_classes = len(self.categories_map)
        self.class_indices = {i: np.argwhere(labels == i).reshape(-1) for i in range(self.num_classes)}

        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def _process_mat_file(self, categories):
        """
        处理 MAT文件并准备数据集。
        """
        mat_data = sio.loadmat(self.data_path)

        total_samples = sum(mat_data[category].shape[0] for category in categories)
        data_shape = list(mat_data[categories[0]].shape)
        data_shape[0] = total_samples
        data_shape = tuple(data_shape)

        all_data = np.zeros(data_shape, dtype=float)
        all_labels = np.zeros(total_samples, dtype=int)

        self._populate_data_and_labels(mat_data, categories, all_data, all_labels)
        categories_map = {i: category for i, category in enumerate(categories)}

        return all_data, all_labels, categories_map

    @staticmethod
    def _populate_data_and_labels(mat_data, categories, all_data, all_labels):
        """
        用MAT文件中的数据填充all_data和all_labels数组。
        """
        current_index = 0
        for i, category in enumerate(categories):
            category_data = mat_data[category]
            num_samples = category_data.shape[0]
            all_data[current_index:current_index + num_samples] = category_data
            all_labels[current_index:current_index + num_samples] = i
            current_index += num_samples

    def get_categories_map(self, label):
        return self.categories_map[label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def split_for_maml(self, train_size=0.7, test_size=0.3):
        assert train_size + test_size == 1., "train + test 的总和必须等于1"

        stratified_indices = []

        for i in range(self.num_classes):
            class_indices = self.class_indices[i]
            np.random.shuffle(class_indices)  # 打乱索引以确保随机性
            num_samples = len(class_indices)

            train_end = int(train_size * num_samples)

            stratified_indices.append((class_indices[:train_end], class_indices[train_end:]))

        train_indices = np.concatenate([stratified_indices[i][0] for i in range(self.num_classes)])
        test_indices = np.concatenate([stratified_indices[i][1] for i in range(self.num_classes)])

        train_dataset = MAMLMatDataset(self, train_indices)
        test_dataset = MAMLMatDataset(self, test_indices)

        return train_dataset, test_dataset

    def split(self, train_size=0.7, test_size=0.3):
        assert train_size + test_size == 1., "train + test 的总和必须等于1"

        stratified_indices = []

        for i in range(self.num_classes):
            class_indices = self.class_indices[i]
            np.random.shuffle(class_indices)  # 打乱索引以确保随机性
            num_samples = len(class_indices)

            train_end = int(train_size * num_samples)

            stratified_indices.append((class_indices[:train_end], class_indices[train_end:]))

        train_indices = np.concatenate([stratified_indices[i][0] for i in range(self.num_classes)])
        test_indices = np.concatenate([stratified_indices[i][1] for i in range(self.num_classes)])

        train_dataset = SubDataset(self.data[train_indices], self.labels[train_indices], self.num_classes)
        test_dataset = SubDataset(self.data[test_indices], self.labels[test_indices], self.num_classes)

        return train_dataset, test_dataset


class MAMLMatDataset(Dataset):
    def __init__(self, parent_dataset, indices):
        self.data = parent_dataset.data[indices]
        self.labels = parent_dataset.labels[indices]
        self.categories_map = parent_dataset.categories_map
        self.num_classes = parent_dataset.num_classes
        self.batch_size = parent_dataset.num_tasks
        self.n_way = parent_dataset.n_way
        self.k_shot = parent_dataset.k_shot
        self.q_query = parent_dataset.q_query

        self.class_indices = {i: np.argwhere(self.labels == i).reshape(-1) for i in range(self.num_classes)}

    def __len__(self):
        return self.batch_size * 8

    def __getitem__(self, i):
        def create_data_samples(classes, k_shot, q_query):
            support = []
            query = []
            for label, class_id in enumerate(classes):
                class_indices = self.class_indices[class_id]
                samples = random.sample(class_indices.tolist(), k_shot + q_query)
                for index in samples[:k_shot]:
                    support.append((self.data[index], label))
                for index in samples[k_shot:]:
                    query.append((self.data[index], label))
            return support, query

        def separate_images_and_labels(data):
            images = [item[0] for item in data]
            labels = [item[1] for item in data]
            return torch.stack(images), torch.tensor(labels, dtype=torch.long)

        selected_classes = list(range(self.num_classes))
        # selected_classes = random.sample(range(self.num_classes), self.n_way)
        support_data, query_data = create_data_samples(selected_classes, self.k_shot, self.q_query)

        random.shuffle(support_data)
        random.shuffle(query_data)

        support_images, support_labels = separate_images_and_labels(support_data)
        query_images, query_labels = separate_images_and_labels(query_data)

        return support_images, support_labels, query_images, query_labels


class SubDataset(Dataset):
    def __init__(self, data, labels, num_classes):
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.class_indices = {i: np.argwhere(self.labels == i).reshape(-1) for i in range(self.num_classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = BaseMatDataset('D:\Code\project1\datasets\mat_datasets\PPMI.mat', [])