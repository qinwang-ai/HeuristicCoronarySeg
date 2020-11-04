import numpy as np
import time
import os
import SimpleITK as sitk
import numba
import utils
import sys
sys.path.append("..")
root = 'xxx'


@numba.jit(nopython=True, fastmath=True)
def npmin2(X, axis=0):
    if axis == 0:
        _min = np.empty(X.shape[1])
        for i in range(X.shape[1]):
            _min[i] = np.min(X[:, i])
    elif axis == 1:
        _min = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            _min[i] = np.min(X[i, :])
    return _min


@numba.jit(nopython=True, fastmath=True)
def npmax2(X, axis=0):
    if axis == 0:
        _min = np.empty(X.shape[1])
        for i in range(X.shape[1]):
            _min[i] = np.max(X[:, i])
    elif axis == 1:
        _min = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            _min[i] = np.max(X[i, :])
    return _min


@numba.jit(nopython=True, fastmath=True)
def within_bbox(x, y, bbox):
    p1 = bbox[0, :]
    p2 = bbox[1, :]
    return p1[0] < x < p2[0] and p1[1] < y < p2[1]


@numba.jit(nopython=True, fastmath=True)
def coron2ht(mask_array, ht_array, buffer):
    points = np.argwhere(mask_array == 1)
    minx, miny, minz = npmin2(points, axis=0)
    maxx, maxy, maxz = npmax2(points, axis=0)

    buff_x, buff_y, buff_z = buffer

    for i in range(minx, maxx + 1):
        for j in range(miny, maxy + 1):
            for k in range(minz, maxz + 1):
                if ht_array[i, j, k] == 0:
                    if within_bbox(j, k, buff_x[i]) or within_bbox(i, k, buff_y[j]) or within_bbox(i, j, buff_z[k]):
                        ht_array[i, j, k] = 1
    return ht_array


def get_init_ht_array(mask_array):
    points = np.argwhere(mask_array == 1)
    cp = points.mean(axis=0).astype(np.int)
    dist = (points - cp)
    dist = (dist ** 2).sum(axis=1)
    di = np.argmin(dist)
    near_p = points[di]
    r = np.abs(cp - near_p)
    ht_array[cp[0] - r[0]:cp[0] + r[0], cp[1] - r[1]:cp[1] + r[1], cp[2] - r[2]:cp[2] + r[2]] = 1
    return ht_array


# calculate bbox
def get_buffer(mask_array):
    points = np.argwhere(mask_array == 1)
    minx, miny, minz = npmin2(points, axis=0).astype(np.int)
    maxx, maxy, maxz = npmax2(points, axis=0).astype(np.int)
    buff_x = np.zeros((mask_array.shape[0], 2, 2))
    buff_y = np.zeros((mask_array.shape[1], 2, 2))
    buff_z = np.zeros((mask_array.shape[2], 2, 2))
    for i in range(minx, maxx + 1):
        points = np.argwhere(mask_array[i, :, :] == 1)
        p1 = npmin2(points, axis=0)
        p2 = npmax2(points, axis=0)
        buff_x[i] = np.array((p1, p2))

    for j in range(miny, maxy + 1):
        points = np.argwhere(mask_array[:, j, :] == 1)
        p1 = npmin2(points, axis=0)
        p2 = npmax2(points, axis=0)
        buff_y[j] = np.array((p1, p2))

    for k in range(minz, maxz + 1):
        points = np.argwhere(mask_array[:, :, k] == 1)
        if len(points) == 0:
            points = np.array([[0, 0]])
            print('pls check this ht nii')
        p1 = npmin2(points, axis=0)
        p2 = npmax2(points, axis=0)
        buff_z[k] = np.array((p1, p2))

    return (buff_x, buff_y, buff_z)


def dilation(ht_array):
    from skimage import morphology
    from skimage.filters import gaussian

    structure = np.zeros((3,3,3))
    structure[1, 1, :] = 1
    for i in range(60):
        ht_array = morphology.binary_dilation(ht_array, selem=structure)
        ht_array[ct_array < 10] = 0

    return ht_array


if __name__ == '__main__':
    f = open(root + 'ct2tp/data_perpare/val.txt')
    file_list = f.readlines()

    for index in range(len(file_list)):
        ct_path = file_list[index].replace('\n', '')
        gt_path = ct_path.replace('CT', 'GT')
        ht_path = ct_path.replace('CT', 'HT')
        if os.path.exists(ht_path):
            print('%s already exists'% ht_path)
            continue

        ct = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct)

        maskHandler = sitk.ReadImage(gt_path)
        mask_array = sitk.GetArrayFromImage(maskHandler)

        # mask2cl
        start = time.time()
        ht_array = np.zeros(mask_array.shape)
        ht_array[mask_array == 1] = 1
        buffer = get_buffer(mask_array)
        ht_array = coron2ht(mask_array, ht_array, buffer)
        ht_array[ct_array < 10] = 0
        ht_array = utils.get_large_single_component(ht_array)
        ht_array = dilation(ht_array)
        print('per nii time used: %0.2f mins' % ((time.time() - start) / 60), ht_path)

        # save
        ht_handler = sitk.GetImageFromArray(ht_array.astype(np.int8))
        ht_handler.SetDirection(maskHandler.GetDirection())
        ht_handler.SetOrigin(maskHandler.GetOrigin())
        ht_handler.SetSpacing(maskHandler.GetSpacing())
        sitk.WriteImage(ht_handler, ht_path)
