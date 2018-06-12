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
