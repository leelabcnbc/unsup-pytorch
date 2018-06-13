"""this file prepares the data for the 2010 NIPS paper
of conv PSD

Koray Kavukcuoglu, Pierre Sermanet, Y-Lan Boureau, Karol Gregor, Michaël Mathieu, Yann LeCun:
Learning Convolutional Feature Hierarchies for Visual Recognition. NIPS 2010: 1090-1098

I will prepare 1000000 25x25 patches, which should be sufficient.
"""
import os
import numpy as np
import h5py

if __name__ == '__main__':
    # save as npy
    with h5py.File(os.path.join(os.path.split(__file__)[0], 'data.hdf5'), 'r') as f:
        (raw_data, data, idx_img, idx_r, idx_c) = (
            f['raw_data'][...], f['data'][...], f['idx_img'][...], f['idx_r'][...], f['idx_c'][...]
        )
        assert raw_data.shape == (5000, 56, 56)
        assert data.shape == (1000000, 1, 25, 25)
        assert idx_img.shape == (1000000,) == idx_r.shape == idx_c.shape
        print(idx_img.min(), idx_img.max())
        print(idx_r.min(), idx_r.max())
        print(idx_c.min(), idx_c.max())
        for idx_global, (data_this, idx_img_this, idx_r_this, idx_c_this) in enumerate(zip(
                data, idx_img, idx_r, idx_c
        )):
            if idx_global % 10000 == 0:
                print(idx_global)
            data_this_ref = raw_data[idx_img_this, idx_r_this:idx_r_this + 25,
                            idx_c_this:idx_c_this + 25]
            data_this_ref = data_this_ref[np.newaxis]
            assert np.array_equal(data_this, data_this_ref.astype(np.float32))
            assert np.std(data_this_ref, ddof=1) > 0.2
