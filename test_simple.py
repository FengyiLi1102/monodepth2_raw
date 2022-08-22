# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import re
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import cv2

import networks
from evaluate_depth import STEREO_SCALE_FACTOR
from layers import disp_to_depth


def parse_args(weight_n=0, model_path="", log_dir=r"/vol/bitbucket/fl4718/monodepth2/assets/output", model_name="",
               min_depth=100.0, max_depth=8000.0):
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default=r"splits/test_files.txt")
                        # default=r"masked_test_1_v")
                        # default=r"test.png")
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        default=model_name)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder',
                        default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true',
                        default=False)
    parser.add_argument("--log_dir",
                        type=str,
                        default=log_dir)
    parser.add_argument("--output_dir",
                        type=str,
                        default=os.path.join(log_dir, f"test_results_{weight_n}/"))
    parser.add_argument("--weight_num",
                        type=int,
                        default=weight_n)
    parser.add_argument("--model_path",
                        type=str,
                        default=model_path)
    parser.add_argument("--min_depth",
                        type=float,
                        default=min_depth)
    parser.add_argument("--max_depth",
                        type=float,
                        default=max_depth)

    return parser.parse_args()


def num_sort(input):
    return list(map(int, re.findall(r"\d+", input)))


def test_simple(args):
    imgL = cv2.imread(
        r"/Users/fyli/Documents/Msc_Computing/Individual_project/Utils/rectified_rendered_data/Left/rgb_0464.PNG")
    imgR = cv2.imread(
        r"/Users/fyli/Documents/Msc_Computing/Individual_project/Utils/rectified_rendered_data/Right/rgb_0464.PNG")
    img_conc = np.concatenate((imgL, imgR), axis=2)
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    # download_model_if_doesnt_exist(args.model_name)
    # Load the model
    # model_paths = glob.glob(os.path.join(args.log_dir, args.model_name, "models/*/"))
    # model_paths.sort(key=num_sort)
    # model_path = model_paths[-1]

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # Only for test
    model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_directory = args.output_dir

    if args.image_path.split(".")[-1] == "txt":
        with open(args.image_path, "r") as tf:
            contents = tf.readlines()
            root_dir = r"/Users/fyli/Documents/Msc_Computing/Individual_project/Utils/rectified_rendered_data"
            paths = [os.path.join(root_dir, line.split(" ")[0], f"rgb_{line.split(' ')[1]}.PNG") for line in contents]
    elif os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            # input_image = pil.open(image_path).convert('RGB')
            # original_width, original_height = input_image.size
            # input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = img_conc
            original_width, original_height = input_image.shape[:2]
            # input_image = input_image.resize((640, 480), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)


            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth_{}.npy".format(output_name, image_path.split("/")[-2]))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                print(type(metric_depth))
                print(metric_depth)
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp_{}.npy".format(output_name, image_path.split("/")[-2]))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp_{}.jpeg".format(output_name, image_path.split("/")[-2]))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    imgL = cv2.imread(r"/Users/fyli/Documents/Msc_Computing/Individual_project/Utils/rectified_rendered_data/Left/rgb_0464.PNG")
    imgR = cv2.imread(r"/Users/fyli/Documents/Msc_Computing/Individual_project/Utils/rectified_rendered_data/Right/rgb_0464.PNG")
    img_conc = np.concatenate((imgL, imgR), axis=2)
    # cv2.imwrite("test.png", img_conc)
    args = parse_args(model_path="models/models_rendered_non_masked/weights_41_0.021649",
                      weight_n=41,
                      log_dir=os.path.dirname(__file__),
                      model_name="local pc")
    test_simple(args)

    # i = 0
    #
    # for model in glob.glob("assets/output/stereo_640x480/model_rendered_masked_0.1_1000/*/"):
    #     print(model)
    #     args = parse_args(weight_n=i, model_path=model)
    #     test_simple(args)
    #     i += 1
