"""this file prepares the data for the 2010 NIPS paper
of conv PSD

Koray Kavukcuoglu, Pierre Sermanet, Y-Lan Boureau, Karol Gregor, Michaël Mathieu, Yann LeCun:
Learning Convolutional Feature Hierarchies for Visual Recognition. NIPS 2010: 1090-1098

I will prepare 1000000 25x25 patches, which should be sufficient.
"""
import os
import numpy as np
import h5py

from torch.utils.serialization.read_lua_file import load_lua

from unsup import dir_dictionary


def load_raw_data():
    raw_data = load_lua(os.path.join(dir_dictionary['debug_reference'],
                                     'tr-berkeley-N5K-M56x56-lcn.bin'))
    raw_data = raw_data.numpy()
    return raw_data


def sample_from_raw_data(std_threshold=0.2, seed=0, ddof=1,
                         num_im=1000000):
    # this ddof stuff really should not matter.
    # here I just want to follow what's done in the original code as much as possible.
    pass
    raw_data = load_raw_data()
    assert raw_data.shape == (5000, 56, 56)
    rng_state = np.random.RandomState(seed=seed)
    # for loop
    collected = 0
    all_imgs = []
    all_img_idx = []
    all_r_idx = []
    all_c_idx = []
    while collected < num_im:
        if collected % 10000 == 0:
            print(collected)
        # randomly select a image
        im_idx = rng_state.randint(5000)
        # then randomly select a patch
        r_idx, c_idx = rng_state.randint(56 - 25 + 1, size=(2,))
        im_candidate = raw_data[im_idx, np.newaxis, r_idx:r_idx + 25, c_idx:c_idx + 25]
        if np.std(im_candidate, ddof=ddof) <= std_threshold:
            continue
        else:
            collected += 1
            # save as float to save space
            all_imgs.append(im_candidate.astype(np.float32))
            all_img_idx.append(im_idx)
            all_r_idx.append(r_idx)
            all_c_idx.append(c_idx)
    return {
        'raw_data': raw_data,
        'data': np.asarray(all_imgs),
        'idx_img': np.asarray(all_img_idx),
        'idx_r': np.asarray(all_r_idx),
        'idx_c': np.asarray(all_c_idx),
    }


if __name__ == '__main__':
    data_dict = sample_from_raw_data()
    # save as npy
    with h5py.File(os.path.join(os.path.split(__file__)[0], 'data.hdf5')) as f:
        if 'data' not in f:
            # 2.4G vs 2.2G. not worth it.
            # f.create_dataset('data', data=a, compression='gzip')
            for k, v in data_dict.items():
                print(k, v.shape)
                f.create_dataset(k, data=v)
