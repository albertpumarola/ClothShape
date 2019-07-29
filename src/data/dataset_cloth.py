import os.path
from src.data.dataset import DatasetBase
import numpy as np
from src.utils import cv_utils
import csv

class DatsetCloth(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(DatsetCloth, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'cloth'

        # init meta
        self._init_meta(opt)

        # read dataset
        self._read_dataset()

    def _init_meta(self, opt):
        self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._rgb_folder = opt[self._name]["rgb_folder"]
        self._depth_folder = opt[self._name]["depth_folder"]
        self._target_file = opt[self._name]["targets_file"]

        if self._is_for == "train":
            self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # get data
        id = self._ids[index]
        img, img_path = self._get_img_by_id(id)
        depth, depth_path = self._get_depth_by_id(id)
        target = self._get_target_by_id(id)

        # pack data
        sample = {'img': img, 'depth': depth, 'target': target}

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        # read ids
        use_ids_filepath = os.path.join(self._root, self._ids_filename)
        self._ids = self._read_valid_ids(use_ids_filepath)

        # read targets
        self._targets = self._read_targets(os.path.join(self._root, self._target_file))

        # read data
        self._imgs_dir = os.path.join(self._root, self._rgb_folder)
        self._depths_dir = os.path.join(self._root, self._depth_folder)

        # dataset size
        self._dataset_size = len(self._ids)

    def _read_valid_ids(self, file_path):
        ids = np.loadtxt(file_path, dtype=np.str)
        return np.expand_dims(ids, 0) if ids.ndim == 0 else ids

    def _read_targets(self, file_path):
        with open(file_path, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=' ')
            dict = {rows[0]:int(rows[1]) for rows in reader}
        return dict

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id+'.jpg')
        return cv_utils.read_cv2_img(filepath), filepath

    def _get_depth_by_id(self, id):
        filepath = os.path.join(self._depths_dir, id+'.jpg')
        return cv_utils.read_cv2_depth(filepath), filepath

    def _get_target_by_id(self, id):
        return self._targets[id]
