
import os
from time import time
import numpy as np
import SimpleITK as sitk

import torch
from net.HeartSegUNet import net
import utils

batch_size = 1
edge = (8, 160, 160)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def eval(patches):
    with torch.no_grad():
        input_array = torch.tensor(np.array(patches)).float()
        preds = net(input_array.cuda())
    return preds

def save_nii(hander, arr, ct_path):
    arr = arr.numpy()
    arr = utils.get_large_single_component(arr)

    nii = sitk.GetImageFromArray(arr.astype(np.int8))
    nii.SetDirection(hander.GetDirection())
    nii.SetOrigin(hander.GetOrigin())
    nii.SetSpacing(hander.GetSpacing())
    save_path = ct_path.replace('CT', 'HT')
    sitk.WriteImage(nii, save_path)


def update_ans(positions, preds, ans):
    ind = 0

    for (xi, yi, zi) in positions:
        ans[:, xi - edge[0]:xi + edge[0], yi - edge[1]:yi + edge[1], zi - edge[2]:zi + edge[2]] += preds[
            ind].cpu().detach()
        ind += 1
    return ans

def test_ht(net):
    import glob
    file_list = list(map(lambda x:x.strip(), open("./valid.txt", 'r').readlines()))
    net.eval()
    print('num of files', len(file_list))
    for index in range(len(file_list)):
        start = time()
        ct_path = file_list[index].replace('\n', '')
        fr_path = ct_path.replace('CT', 'FR')
        save_path = ct_path.replace('CT', 'HT')

        ct = sitk.ReadImage(ct_path)
        fr = sitk.ReadImage(fr_path)
        ct_array = sitk.GetArrayFromImage(ct)
        fr_array = sitk.GetArrayFromImage(fr)

        # loop patchs
        ans = torch.zeros((2,) + ct_array.shape)
        patches = []
        gts = []
        positions = []
        gap = 2
        ind = 0
        x, y, z = edge
        sx, sy, sz = ct_array.shape
        while True:

            patch = ct_array[x - edge[0]:x + edge[0], y - edge[1]:y + edge[1], z - edge[2]:z + edge[2]]
            patch_fr = fr_array[x - edge[0]:x + edge[0], y - edge[1]:y + edge[1], z - edge[2]:z + edge[2]]

            new_patch = [patch, patch_fr]

            # save in arrays
            patches.append(new_patch)
            positions.append((x, y, z))

            # eval
            if len(patches) >= batch_size:
                preds = eval(patches)
                ans = update_ans(positions, preds, ans)
                patches, positions, gts = [], [], []

            # switch x y z
            if x < sx - edge[0]:
                x += edge[0] * gap
            elif y < sy - edge[1]:
                y += edge[1] * gap
                x = edge[0]
            elif z < sz - edge[2]:
                z += edge[2] * gap
                x = edge[0]
                y = edge[1]
            else:
                break

            # prevent boundary
            if x > sx - edge[0]:
                x = sx - edge[0]
            if y > sy - edge[1]:
                y = sy - edge[1]
            if z > sz - edge[2]:
                z = sz - edge[2]

            # end condition

        if len(patches) > 0:
            preds = eval(patches)
            ans = update_ans(positions, preds, ans)
            patches, positions, gts = [], [], []

        file_name = file_list[index].split('/')[-2]

        ans = torch.argmax(ans, dim=0)
        save_nii(ct, ans, ct_path)

        print('time:{:.3f} min'.format((time() - start) / 60), file_name)

if __name__ == '__main__':
    pretrained_model_path = "./module/best_ct2ht.pth"
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(pretrained_model_path), strict=False)
    test_ht(net)

