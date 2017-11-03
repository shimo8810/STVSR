import numpy as np
import h5py

outf = h5py.File('hoge.hdf5', 'w')
dset = outf.create_dataset('data',shape=(10,3,31,31), maxshape=(None, 3, 31,31), chunks=True)
size = 0
for i in range(10):
    print(size, size + 10)
    data = np.random.rand(10, 3, 31,31)
    print(data)
    dset.resize((size + 10, 3, 31,31))
    dset[size : size + 10, :, :, :] = data
    outf.flush()
    size = size + 10
outf.flush()
outf.close()
