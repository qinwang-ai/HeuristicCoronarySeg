import numpy as np
import time
import os
import SimpleITK as sitk
from skimage.morphology import skeletonize_3d
import glob
root = '/home/qinwang/miccai/'


def mask2cl(mask_handler, file):
    start = time.time()
    mask = sitk.GetArrayFromImage(mask_handler)

    # get cl
    output_volume = skeletonize_3d(mask.astype(np.int8))
    output_volume[output_volume == 2] = 1

    cl_handler = sitk.GetImageFromArray(output_volume.astype(np.int8))
    cl_handler.SetDirection(mask_handler.GetDirection())
    cl_handler.SetOrigin(mask_handler.GetOrigin())
    cl_handler.SetSpacing(mask_handler.GetSpacing())

    stop = time.time()
    elapsed = stop - start
    print('per nii time used: %0.2f mins' % (elapsed / 60), file, np.unique(output_volume))
    return cl_handler


if __name__ == '__main__':
    file_list = glob.glob('./RotterdamCoronaryDataset/*/CT.nii.gz')
    # file_list = ['/home/student/data/qinwang/ct_dataset/RotterLumen/dataset16/CTsp823.nii.gz']

    for index in range(len(file_list)):
        ct_path = file_list[index].replace('\n', '')
        # gt_path = ct_path.replace('CT', 'CR')
        gt_path = ct_path.replace('CT', 'TP')
        cl_path = ct_path.replace('CT', 'CL')
        # if os.path.exists(cl_path):
        #     print('exists skip')
        #     continue

        maskHandler = sitk.ReadImage(gt_path)

        # mask2cl
        clHandler = mask2cl(maskHandler, gt_path)

        # write
        sitk.WriteImage(clHandler, cl_path)
        print(cl_path, 'saved')


