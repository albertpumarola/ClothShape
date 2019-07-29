#!/usr/bin/python
import os
from tqdm import tqdm
import scipy.io
import argparse
import threading
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_img_dir', type=str, default='/home/apumarola/code/phd/ClothShape/datasets/cloth/original/depth', help='Original input images directory')
parser.add_argument('-s', '--desired_size', type=int, default=128, help='Desires output size')
parser.add_argument('-cmin', '--crop_min', type=int, default=0, help='Desires min crop position')
parser.add_argument('-cmax', '--crop_max', type=int, default=512, help='Desires max crop position')
parser.add_argument('-oi', '--ouput_img_dir', type=str, default='/home/apumarola/code/phd/ClothShape/datasets/cloth/processed/depth', help='Directory to store processed images')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Num workers')
args = parser.parse_args()


class CropDataset:
    def _get_all_files_in_subfolders(self, dir, filter):

        filepaths = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        is_obj = lambda path: any(path.endswith(extension) for extension in filter)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_obj(fname):
                    path = os.path.join(root, fname)
                    path = os.path.relpath(path, dir)
                    filepaths.append(path)

        filepaths.sort()
        return filepaths

    def _save_img(self, img, output_path):
        ouput_dir = os.path.dirname(output_path)
        if not os.path.isdir(ouput_dir):
            os.makedirs(ouput_dir)
        cv2.imwrite(output_path, img)

    def _crop(self, img_subfilepaths, do_print=False):
        for img_subfilepath in tqdm(img_subfilepaths) if do_print else img_subfilepaths:
            img_path = os.path.join(args.input_img_dir, img_subfilepath)

            crop_img = cv2.imread(img_path)
            height, width = crop_img.shape[0], crop_img.shape[1]
            min_i = self._clip(args.crop_min, height)
            max_i = self._clip(args.crop_max, height)
            min_j = self._clip(args.crop_min, width)
            max_j = self._clip(args.crop_max, width)

            crop_img = crop_img[min_i:max_i, min_j:max_j, :]
            crop_img = cv2.resize(crop_img, (args.desired_size, args.desired_size), interpolation=cv2.INTER_AREA)

            output_path = os.path.join(args.ouput_img_dir, img_subfilepath[:-4]+'.jpg')
            self._save_img(crop_img, output_path)

    def _clip(self, i, size):
        return np.clip(i, 0, size)

    def crop(self):
        img_subfilepaths = self._get_all_files_in_subfolders(args.input_img_dir, ['.jpg', '.png'])
        files_per_thread = np.array_split(np.array(img_subfilepaths), args.num_workers)

        threads = []
        for i in range(args.num_workers):
            do_print = (i == 0)
            t = threading.Thread(target=self._crop, args=(files_per_thread[i], do_print,))
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()


def main():
    crop_tool = CropDataset()
    crop_tool.crop()

if __name__ == '__main__':
    main()