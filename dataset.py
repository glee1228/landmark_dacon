from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
import pandas as pd
import os
from tqdm import tqdm
import random


class LandmarkDataset(Dataset):
    def __init__(self, base_dir, data_csv, category_csv, transform=None, phase='train', seed=0):
        self.base_dir = base_dir
        self.category = [i[1] for i in pd.read_csv(category_csv).values.tolist()]

        self.region = {c: region for region in os.listdir(base_dir) for c in os.listdir(os.path.join(base_dir, region))}
        self.samples = self.split_samples(data_csv, phase, seed)

        self.loader = default_loader
        self.transform = transform if transform else transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, cid = self.samples[idx]
        label = self.category[cid]
        region = self.region[label]


        path = os.path.join(self.base_dir, region, label, iid + '.JPG')

        img = self.transform(self.loader(path))
        return img, iid, cid, label

    def split_samples(self, data_csv, phase, seed):
        samples_per_class = defaultdict(list)
        for iid, cid in pd.read_csv(data_csv).values.tolist():
            samples_per_class[cid].append([iid, cid])
        samples = []
        random.seed(seed)
        for k, v in samples_per_class.items():
            random.shuffle(v)
            c = int(len(v) * 0.8)
            l = v[:c] if phase.lower() == 'train' else v[c:]
            samples.extend(l)
        return samples


class TestDataset(Dataset):
    def __init__(self, base_dir, submission_csv, category_csv, transform=None):
        self.base_dir = base_dir
        self.cateogory = [i[1] for i in pd.read_csv(category_csv).values.tolist()]
        self.test = pd.read_csv(submission_csv)['id'].values.tolist()

        self.loader = default_loader
        self.transform = transform if transform else transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        iid = self.test[idx]
        path = os.path.join(self.base_dir, iid[0], iid + '.JPG')
        img = self.transform(self.loader(path))
        return img, iid


if __name__ == '__main__':
    train_dir = '/data/public/train'
    train_csv = '/data/public/train2.csv'
    val_csv = '/data/public/val2.csv'
    test_dir = '/data/public/test'
    submission = '/data/public/sample_submission.csv'

    category_csv = '/data/public/category.csv'

    train_dataset = LandmarkDataset(train_dir, train_csv, category_csv, transform=None, phase='train')

    valid_dataset = LandmarkDataset(train_dir, val_csv, category_csv, transform=None, phase='valid')

    test_dataset = TestDataset(test_dir, submission, category_csv)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    with tqdm(train_loader) as pbar:
        for i in pbar:
            pbar.write(str(i[1][:5]))

    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)
    with tqdm(valid_loader) as pbar:
        for i in pbar:
            pbar.write(str(i[1][:5]))

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    with tqdm(test_loader) as pbar:
        for i in pbar:
            pbar.write(str(i[1][:5]))
