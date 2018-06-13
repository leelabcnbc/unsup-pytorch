"""this file will make sure the data generated in _data.npy can be reproduced in python"""

# TODO: when actually writing my own loader, I need to make sure to compute std (n vs n-1;
# torch I believe uses n-1), and check that it's above some threshold (0.2 in the original code).

# the indices of images and cropping locations are obtained by loading `demo_data_mod.lua` instead
# of `demo_data.lua` in any of demo scripts.

from torch.utils.serialization.read_lua_file import load_lua
import os
import numpy as np


def check_no_conv():
    # all 1-indexed
    loc_info = [
        (3786, 3, 45),
        (2801, 22, 21),
        (2748, 26, 4),
        (2511, 38, 11),
        (1902, 16, 46),
        (296, 36, 37),
        (1067, 45, 17),
        (4928, 6, 8),
        (1459, 32, 15),
        (1230, 34, 42)
    ]

    # load data
    raw_img_file = os.path.join(os.path.split(__file__)[0], '..', 'reference',
                                'tr-berkeley-N5K-M56x56-lcn.bin')
    raw_img_all = load_lua(raw_img_file).numpy()
    imgs = []
    for img_idx, row_idx, col_idx in loc_info:
        imgs.append(raw_img_all[img_idx - 1, row_idx - 1:row_idx - 1 + 9,
                    col_idx - 1:col_idx - 1 + 9].ravel())
    imgs = np.asarray(imgs)
    print(np.std(imgs, axis=1, ddof=1))
    # original output
    # Linear sparse coding
    # Image 3786 ri= 3 ci= 45 std= 0.78800831645529
    # -2.2322928905487	1.7002385854721
    # Image 2801 ri= 22 ci= 21 std= 0.51567251698965
    # -1.1077057123184	1.2338403463364
    # Image 2748 ri= 26 ci= 4 std= 0.6950226585778
    # -0.8757591843605	1.5443687438965
    # Image 2511 ri= 38 ci= 11 std= 0.21036643892944
    # -0.46495559811592	0.3957097530365
    # Image 1902 ri= 16 ci= 46 std= 0.25498365290578
    # -0.76834791898727	0.59102118015289
    # Image 296 ri= 36 ci= 37 std= 0.20336900751392
    # -0.54546678066254	0.46946200728416
    # Image 1067 ri= 45 ci= 17 std= 0.37703898142497
    # -2.210765838623	0.81278049945831
    # Image 2652 ri= 3 ci= 7 std= 0.044280564610565
    # Image 4928 ri= 6 ci= 8 std= 0.72752323124476
    # -1.9195595979691	1.4983677864075
    # Image 1459 ri= 32 ci= 15 std= 0.66576965972612
    # -1.0815800428391	2.6140785217285
    # Image 1564 ri= 44 ci= 13 std= 0.067697619577861
    # Image 4662 ri= 29 ci= 13 std= 0.05125079460947
    # Image 1230 ri= 34 ci= 42 std= 0.58343760521117
    # -1.3921706676483	1.3081703186035

    # compare with other
    imgs_ref = np.load(os.path.join(os.path.split(__file__)[0], '..', 'reference',
                                    'demo_fista_debug_data.npy'))
    assert np.array_equal(imgs, imgs_ref)
    print(imgs.shape)


def check_conv():
    # all 1-indexed
    loc_info = [
        (4576, 2, 6),
        (68, 10, 25),
        (3500, 8, 20),
        (3148, 12, 8),
        (288, 28, 18),
        (265, 3, 14),
        (4162, 30, 21),
        (1029, 23, 24),
        (2749, 28, 7),
        (2782, 25, 25),
    ]

    # load data
    raw_img_file = os.path.join(os.path.split(__file__)[0], '..', 'reference',
                                'tr-berkeley-N5K-M56x56-lcn.bin')
    raw_img_all = load_lua(raw_img_file).numpy()
    imgs = []
    for img_idx, row_idx, col_idx in loc_info:
        imgs.append(raw_img_all[img_idx - 1, row_idx - 1:row_idx - 1 + 25,
                    col_idx - 1:col_idx - 1 + 25][np.newaxis])
    imgs = np.asarray(imgs)
    print(np.std(imgs, axis=(1, 2, 3), ddof=1))
    # original output
    # Image 4576 ri= 2 ci= 6 std= 0.55018627982933
    # -1.7741584777832	1.8689628839493
    # Image 1678 ri= 31 ci= 23 std= 0.076197773575514
    # Image 68 ri= 10 ci= 25 std= 0.68954769256738
    # -2.3640785217285	2.2259881496429
    # Image 3500 ri= 8 ci= 20 std= 0.43323739124233
    # -1.5980268716812	2.252289056778
    # Image 3131 ri= 24 ci= 3 std= 0
    # Image 3148 ri= 12 ci= 8 std= 0.29113789034902
    # -1.8220963478088	0.71125489473343
    # Image 288 ri= 28 ci= 18 std= 0.77441402903409
    # -2.3599710464478	2.6021389961243
    # Image 265 ri= 3 ci= 14 std= 0.20241389240196
    # -0.70054388046265	0.56267201900482
    # Image 870 ri= 20 ci= 19 std= 0.10149980428585
    # Image 4162 ri= 30 ci= 21 std= 0.24563511886418
    # -0.84212404489517	0.64118164777756
    # Image 3159 ri= 17 ci= 24 std= 0.19618505804854
    # Image 1029 ri= 23 ci= 24 std= 0.21406000155881
    # -1.0242427587509	0.97620660066605
    # Image 2749 ri= 28 ci= 7 std= 0.45778242314877
    # -1.2723920345306	1.7770617008209
    # Image 2782 ri= 25 ci= 25 std= 0.27222106829896
    # -0.84365510940552	1.8664221763611

    # compare with other
    imgs_ref = np.load(os.path.join(os.path.split(__file__)[0], '..', 'reference',
                                    'demo_psd_conv_debug_data.npy'))
    assert np.array_equal(imgs, imgs_ref)
    print(imgs.shape)


if __name__ == '__main__':
    print('no conv')
    check_no_conv()
    print('conv')
    check_conv()
