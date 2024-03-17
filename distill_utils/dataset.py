import os
import numpy as np
import random
from typing import Any
import torch
import torch.utils.data as tdata
import os.path as osp
import csv
from PIL import Image
from torchvision import transforms
import pickle
import functools
import json

NUM_FRAMES = 16
FRAME_GAP = 4

class singleKinetics400(tdata.Dataset):
    def __init__(self, path, split, transform):
        NUM_FRAMES = 8
        self.transform = transform
        self.split = split

        csv_split = "validate" if split == "val" else split
        csv_path = osp.join(path, f"{csv_split}.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()
        item_to_skip = 0

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                assert csv_split == item["split"]
                name = "%s_%06d_%06d" % (
                    item["youtube_id"],
                    int(item["time_start"]),
                    int(item["time_end"]),
                )

                sample_dir = osp.join(path, split, name)
                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    sample_dir = osp.join(path, "replacement", name)

                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    item_to_skip += 1
                else:
                    self.label_strs.append(item["label"])
                    self.class_strs.add(item["label"])
                    self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        print(f"{split}: {len(self.video_dirs)} samples, Skipped {item_to_skip} items")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        length = len(os.listdir(path))
        f = os.listdir(path)[random.randint(0, length-1)]
        p = osp.join(path, f)
        im_pil = Image.open(p)
        im = self.transform(im_pil)
        return im, label

class Kinetics400(tdata.Dataset):
    def __init__(self, path, split, transform):
        if path.split("/")[-1] == "kinetics_64x64x8":
            NUM_FRAMES=8
        else:
            NUM_FRAMES=16
        self.transform = transform
        self.split = split

        csv_split = "validate" if split == "val" else split
        csv_path = osp.join(path, f"{csv_split}.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()
        item_to_skip = 0

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                assert csv_split == item["split"]
                name = "%s_%06d_%06d" % (
                    item["youtube_id"],
                    int(item["time_start"]),
                    int(item["time_end"]),
                )

                sample_dir = osp.join(path, split, name)
                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    sample_dir = osp.join(path, "replacement", name)

                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    item_to_skip += 1
                else:
                    self.label_strs.append(item["label"])
                    self.class_strs.add(item["label"])
                    self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        print(f"{split}: {len(self.video_dirs)} samples, Skipped {item_to_skip} items")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        vid = []
        for f in os.listdir(path):
            p = osp.join(path, f)
            im_pil = Image.open(p)
            im = self.transform(im_pil)
            vid.append(im)
        vid = torch.stack(vid)
        # print(vid.shape)
        return vid, label

class UCF101(tdata.Dataset):
    def __init__(self, path, split, transform=None):
        "Initialization"
        self.data_path = osp.join(path, "jpegs_112")

        begin_frame, end_frame, skip_frame = 1, 24, 3
        self.frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
        

        self.transform = transform
        self.split = split

        csv_path = osp.join(path, "ucf101_splits1.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)
        

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]
        
        print("how many classes:",len(self.class_strs))
        print("UCF101 init finished")

    def __len__(self):
        return len(self.video_dirs)

    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        for i in self.frames:
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )

            if flip:
                image = transforms.functional.hflip(image)

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        if length < NUM_FRAMES * FRAME_GAP:
            skip = length // NUM_FRAMES
        else:
            skip = FRAME_GAP

        if self.start[index] == -1 or self.split == "test":
            self.start[index] = np.random.randint(1, length - (NUM_FRAMES - 1) * skip)   
        else:
            self.start[index] = self.start[index]
        self.frames = np.arange(self.start[index], self.start[index] + NUM_FRAMES * skip, skip).tolist()

        X = self.read_images(
            path, self.transform
        )

        return X, label

    def get_all_frames(self, index):
        X = []
        path = self.video_dirs[index]
        length = len(os.listdir(path))

        for i in range(1, length+1):
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )
            image = self.transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X, length

class HMDB51(tdata.Dataset):
    def __init__(self, path, split, transform):
        self.data_path = osp.join(path, "jpegs_112")

        begin_frame, end_frame, skip_frame = 1, 24, 3
        self.frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
        
        self.transform = transform
        self.split = split
        

        csv_path = osp.join(path, "hmdb51_splits.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]


    def __len__(self):
        return len(self.video_dirs)
    
    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        for i in self.frames:
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )

            if flip:
                image = transforms.functional.hflip(image)

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        if length < NUM_FRAMES * FRAME_GAP:
            skip = length // NUM_FRAMES
        else:
            skip = FRAME_GAP

        if self.start[index] == -1 or self.split == "test":
            self.start[index] = np.random.randint(1, length - (NUM_FRAMES - 1) * skip)   
        else:
            self.start[index] = self.start[index]
        self.frames = np.arange(self.start[index], self.start[index] + NUM_FRAMES * skip, skip).tolist()

        X = self.read_images(
            path, self.transform
        ) 

        return X, label

    def get_all_frames(self, index):
        X = []
        path = self.video_dirs[index]
        length = len(os.listdir(path))

        for i in range(1, length+1):
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )
            image = self.transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X, length

class miniUCF101(tdata.Dataset):
    def __init__(self, path, split, transform=None, sample='random'):
        "Initialization"
        self.data_path = osp.join(path, "jpegs_112")

        begin_frame, end_frame, skip_frame = 1, 24, 3
        self.frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

        self.transform = transform
        self.split = split
        self.sample = sample

        csv_path = osp.join(path, "ucf50_splits1.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)
        

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]
        
        print("how many classes:",len(self.class_strs))
        print("miniUCF init finished")

    def __len__(self):
        return len(self.video_dirs)

    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        for i in self.frames:
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )

            if flip:
                image = transforms.functional.hflip(image)

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        if length < NUM_FRAMES * FRAME_GAP:
            skip = length // NUM_FRAMES
        else:
            skip = FRAME_GAP

        if self.start[index] == -1 or self.split == "test":
            self.start[index] = np.random.randint(1, length - (NUM_FRAMES - 1) * skip)   
        else:
            self.start[index] = self.start[index]
        
        if self.sample == 'random':
            self.frames = np.arange(self.start[index], self.start[index] + NUM_FRAMES * skip, skip).tolist()
        elif self.sample == 'split-random':
            seg_len = length // 16
            seg_starts = [i * seg_len for i in range(16)]
            seg_ends = [(i + 1) * seg_len for i in range(15)] + [length]
            seg_indices = []
            for start, end in zip(seg_starts, seg_ends):
                seg_indices.append(np.random.randint(start, end))
            self.frames = seg_indices
            self.frames = [i+1 for i in self.frames]

        X = self.read_images(
            path, self.transform
        ) 

        return X, label

    def get_all_frames(self, index):
        X = []
        path = self.video_dirs[index]
        length = len(os.listdir(path))

        for i in range(1, length+1):
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )
            image = self.transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X, length
     
class miniHMDB51(tdata.Dataset):
    def __init__(self, path, split, transform):
        self.data_path = osp.join(path, "jpegs_112")

        begin_frame, end_frame, skip_frame = 1, 24, 3
        self.frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
        
        self.transform = transform
        self.split = split
        

        csv_path = osp.join(path, "hmdb25_splits.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]

    def __len__(self):
        return len(self.video_dirs)
    
    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        for i in self.frames:
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )

            if flip:
                image = transforms.functional.hflip(image)

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        if length < NUM_FRAMES * FRAME_GAP:
            skip = length // NUM_FRAMES
        else:
            skip = FRAME_GAP

        if self.start[index] == -1 or self.split == "test":
            self.start[index] = np.random.randint(1, length - (NUM_FRAMES - 1) * skip)   
        else:
            self.start[index] = self.start[index]
        self.frames = np.arange(self.start[index], self.start[index] + NUM_FRAMES * skip, skip).tolist()

        X = self.read_images(
            path, self.transform
        )  

        return X, label

    def get_all_frames(self, index):
        X = []
        path = self.video_dirs[index]
        length = len(os.listdir(path))

        for i in range(1, length+1):
            image = Image.open(
                os.path.join(path, "frame{:06d}.jpg".format(i))
            )
            image = self.transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X, length

class staticHMDB51(tdata.Dataset):
    def __init__(self, path, split, transform, frames=16):
        self.data_path = osp.join(path, "jpegs_112")

        self.start = 1
        self.frames = frames
        
        self.transform = transform
        self.split = split
        

        csv_path = osp.join(path, "hmdb51_splits.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]

    def __len__(self):
        return len(self.video_dirs)
    
    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        image = Image.open(
            os.path.join(path, "frame{:06d}.jpg".format(self.start))
        )

        if flip:
            image = transforms.functional.hflip(image)

        if use_transform is not None:
            image = use_transform(image)

        if self.frames == 1:
            return image
        
        for i in range(self.frames):
            X.append(image)

        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        self.start = np.random.randint(1, length)

        X = self.read_images(
            path, self.transform
        )  # (input) spatial images

        return X, label

class staticUCF101(tdata.Dataset):
    def __init__(self, path, split, transform, frames=16, split_num=1, split_id=0):
        self.data_path = osp.join(path, "jpegs_112")

        self.start = 1
        self.frames = frames
        self.split_num = split_num
        self.split_id = 0 if split_id >= split_num else split_id
        print("split_num:",self.split_num)
        print("split_id:",self.split_id)
        
        self.transform = transform
        self.split = split
        

        csv_path = osp.join(path, "ucf101_splits1.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]

    def __len__(self):
        return len(self.video_dirs)
    
    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        image = Image.open(
            os.path.join(path, "frame{:06d}.jpg".format(self.start))
        )

        if flip:
            image = transforms.functional.hflip(image)

        if use_transform is not None:
            image = use_transform(image)

        if self.frames == 1:
            return image
        
        for i in range(self.frames):
            X.append(image)

        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))

        self.start = np.random.randint(length//self.split_num * self.split_id + 1, length//self.split_num * (self.split_id + 1))

        X = self.read_images(
            path, self.transform
        )  # (input) spatial images

        return X, label

class staticUCF50(tdata.Dataset):
    def __init__(self, path, split, transform, frames=16, split_num=1, split_id=0, split_mode='mean'):
        self.data_path = osp.join(path, "jpegs_112")

        self.start = 1
        self.frames = frames
        self.split_num = split_num
        self.split_id = 0 if split_id >= split_num else split_id
        print("split_num:",self.split_num)
        print("split_id:",self.split_id)
        
        self.transform = transform
        self.split = split
        self.split_mode = split_mode
        print("split_mode:",self.split_mode)
        

        csv_path = osp.join(path, "ucf50_splits1_max.csv")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()
        self.split_lists = []

        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            for item in reader:
                if item["split"] != split:
                    continue
                name = item["folder_name"]
                sample_dir = osp.join(self.data_path, name)
                split_index = item['split_index'].strip('][').split(', ')
                split_index = sorted(split_index, key=lambda i: int(i)) 

                self.label_strs.append(item["label"])
                self.class_strs.add(item["label"])
                self.video_dirs.append(sample_dir)
                self.split_lists.append(split_index)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        length = len(self.video_dirs)
        self.start = [-1 for i in range(length)]

    def __len__(self):
        return len(self.video_dirs)
    
    def read_images(self, path, use_transform):
        X = []
        if random.random() > 0.5:
            flip = True
        else:
            flip = False
            
        image = Image.open(
            os.path.join(path, "frame{:06d}.jpg".format(self.start))
        )

        if flip:
            image = transforms.functional.hflip(image)

        if use_transform is not None:
            image = use_transform(image)

        if self.frames == 1:
            return image
        
        for i in range(self.frames):
            X.append(image)

        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]

        length = len(os.listdir(path))
        
        if self.split_mode == 'mean':
            self.start = np.random.randint(length//self.split_num * self.split_id + 1, length//self.split_num * (self.split_id + 1))
        elif self.split_mode == 'feature':
            if self.split_id == 0:
                self.start = np.random.randint(1, int(self.split_lists[index][0])+1)
            elif self.split_id == 3:
                self.start = np.random.randint(int(self.split_lists[index][2])+1, length)
            else:
                self.start = np.random.randint(int(self.split_lists[index][self.split_id-1])+1, int(self.split_lists[index][self.split_id])+1)
        else:
            print("split_mode error!")
            exit()

        X = self.read_images(
            path, self.transform
        )

        return X, label

class SSv2(tdata.Dataset):
    def __init__(self, path, split, transform):
        if path.split("/")[-1] == "SSv2_64x8":
            NUM_FRAMES=8
        else:
            NUM_FRAMES=16
        self.transform = transform
        self.split = split

        json_path = osp.join(path, f"annot_{split}.json")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()
        item_to_skip = 0

        with open(json_path) as fp:
            data = fp.read()
            content = json.loads(data)
            for item in content:
                name = item['id']

                sample_dir = osp.join(path, 'frame', name)
                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    item_to_skip += 1
                    print("skip", name)
                else:
                    self.label_strs.append(item["class"])
                    self.class_strs.add(item["class"])
                    self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        print(f"{split}: {len(self.video_dirs)} samples, Skipped {item_to_skip} items")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        vid = []
        for f in os.listdir(path):
            p = osp.join(path, f)
            im_pil = Image.open(p)
            im = self.transform(im_pil)
            vid.append(im)
        vid = torch.stack(vid)
        return vid, label

class singleSSv2(tdata.Dataset):
    def __init__(self, path, split, transform):
        NUM_FRAMES = 8
        self.transform = transform
        self.split = split

        json_path = osp.join(path, f"annot_{split}.json")

        self.video_dirs = []
        self.label_strs = []
        self.class_strs = set()
        item_to_skip = 0

        with open(json_path) as fp:
            data = fp.read()
            content = json.loads(data)
            for item in content:
                name = item['id']

                sample_dir = osp.join(path, 'frame', name)
                if (
                    not osp.exists(sample_dir)
                    or len(os.listdir(sample_dir)) != NUM_FRAMES
                ):
                    item_to_skip += 1
                    print("skip", name)
                else:
                    self.label_strs.append(item["class"])
                    self.class_strs.add(item["class"])
                    self.video_dirs.append(sample_dir)

        self.class_strs = sorted(self.class_strs)
        self.class_2_idx = {x: i for i, x in enumerate(self.class_strs)}
        self.labels = [self.class_2_idx[l] for l in self.label_strs]
        self.targets = self.labels  # alias

        print(f"{split}: {len(self.video_dirs)} samples, Skipped {item_to_skip} items")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        length = len(os.listdir(path))
        f = os.listdir(path)[random.randint(0, length-1)]
        p = osp.join(path, f)
        im_pil = Image.open(p)
        im = self.transform(im_pil)
        return im, label


if __name__ == "__main__":
    channel = 3
    im_size = (64, 64) 
    num_classes = 174

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]  # use imagenet transform

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    ssv2 = SSv2(
        "path_for_SSv2","train",transform=transform
    )
    _ssv2 = SSv2[3007]
    print("label:", _ssv2[1])
    print("shape:", _ssv2[0].shape)
