from __future__ import absolute_import, division, print_function

import os
import re

import PIL.Image as pil
import numpy as np

from .mono_dataset import MonoDataset


class CloudDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(CloudDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # College setting
        # f = 5.084752337001038995e+02
        # c_x = 3.225185456035997049e+02
        # c_y = 2.505901000840876520e+02

        # Rendered setting
        f = 530.227
        c_x = 320.0
        c_y = 240

        self.K = np.array([[f / 640, 0,       c_x / 640, 0],
                           [0,       f / 480, c_y / 480, 0],
                           [0,       0,       1,         0],
                           [0,       0,       0,         1]], dtype=np.float32)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, do_flip, rendered=False):
        color = self.loader(self.get_image_path(folder, frame_index, rendered))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class CloudRAWDataset(CloudDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(CloudRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, rendered=False):
        if rendered:
            if folder == "Left":
                f_str = "rgb_{:04d}.PNG".format(frame_index)
            else:
                f_str = "rgb_{:04d}.PNG".format(frame_index)
        else:
            if re.split(r"[/_]+", folder)[1] == "tl4":
                f_str = "img_{}_left.png".format(frame_index)  # img_1_left
            else:
                f_str = "img_{}_right.png".format(frame_index)
        # f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # data_path = dataset/
        # folder = left/tl4_******/
        image_path = os.path.join(self.data_path, folder, f_str)

        return image_path

    # def get_depth(self, folder, frame_index, side, do_flip):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])
    #
    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
    #
    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
    #
    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)
    #
    #     return depth_gt


# class CloudOdomDataset(CloudDataset):
#     """KITTI dataset for odometry training and testing
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(CloudOdomDataset, self).__init__(*args, **kwargs)
#
#     def get_image_path(self, folder, frame_index, side):
#         f_str = "{:06d}{}".format(frame_index, self.img_ext)
#         image_path = os.path.join(
#             self.data_path,
#             "sequences/{:02d}".format(int(folder)),
#             "image_{}".format(self.side_map[side]),
#             f_str)
#         return image_path
#
#
# class CloudDepthDataset(CloudDataset):
#     """KITTI dataset which uses the updated ground truth depth maps
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(CloudDepthDataset, self).__init__(*args, **kwargs)
#
#     def get_image_path(self, folder, frame_index, side):
#         f_str = "{:010d}{}".format(frame_index, self.img_ext)
#         image_path = os.path.join(
#             self.data_path,
#             folder,
#             "image_0{}/data".format(self.side_map[side]),
#             f_str)
#         return image_path
#
#     def get_depth(self, folder, frame_index, side, do_flip):
#         f_str = "{:010d}.png".format(frame_index)
#         depth_path = os.path.join(
#             self.data_path,
#             folder,
#             "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
#             f_str)
#
#         depth_gt = pil.open(depth_path)
#         depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
#         depth_gt = np.array(depth_gt).astype(np.float32) / 256
#
#         if do_flip:
#             depth_gt = np.fliplr(depth_gt)
#
#         return depth_gt


if __name__ == "__main__":
    data_path = r"dataset/"
    filenames = r"filename"
    height = 480
    width = 640
    frame_idxs = [0, -1, 1, "s"]
    num_scales = 4
    demo = CloudRAWDataset(data_path, filenames,
                           height,
                           width,
                           frame_idxs,
                           num_scales,
                           is_train=False,
                           img_ext='.jpg')
