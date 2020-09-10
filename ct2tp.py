import os
from time import time
import numpy as np
import glob
import SimpleITK as sitk
import torch
from net.ResUnetTP import net

edge = (8 // 2, 320 // 2, 320 // 2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 1
Max_ratio = 0.65
pretrained_model_path = "./module/best_ct2tp.pth"

file_list = glob.glob('./RotterdamCoronaryDataset/*/CT.nii.gz')


def eval(patches):
    with torch.no_grad():
        input_array = torch.tensor(np.array(patches)).float()
        preds = net(input_array.cuda())
    return preds


def save_nii(hander, arr, ct_path):
    arr = arr.numpy()

    Max = arr.max()
    threshold = Max * Max_ratio
    arr[arr < threshold] = 0
    arr[arr >= threshold] = 1

    nii = sitk.GetImageFromArray(arr.astype(np.int8))
    nii.SetDirection(hander.GetDirection())
    nii.SetOrigin(hander.GetOrigin())
    nii.SetSpacing(hander.GetSpacing())
    save_path = ct_path.replace('CT', 'TP')
    sitk.WriteImage(nii, save_path)


def update_ans(positions, preds, ans, counter):
    ind = 0
    preds = preds.squeeze(dim=1)

    for (xi, yi, zi) in positions:
        ans[xi - edge[0]:xi + edge[0], yi - edge[1]:yi + edge[1], zi - edge[2]:zi + edge[2]] += preds[
            ind].cpu().detach()
        counter[xi - edge[0]:xi + edge[0], yi - edge[1]:yi + edge[1], zi - edge[2]:zi + edge[2]] += 1
        ind += 1
    return ans, counter


if __name__ == '__main__':
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(pretrained_model_path), strict=False)
    net.eval()

    # 训练网络
    start1 = time()

    for index in range(len(file_list)):
        start = time()
        ct_path = file_list[index].strip()

        fr_path = ct_path.replace('CT', 'FR')
        save_path = ct_path.replace('CT', 'TP')
        if os.path.exists(save_path):
            print(save_path, 'existed skipped')
            continue

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path)
        fr = sitk.ReadImage(fr_path)
        ct_array = sitk.GetArrayFromImage(ct)
        fr_array = sitk.GetArrayFromImage(fr)

        # loop patchs
        ans = torch.zeros(ct_array.shape)
        counter = torch.zeros(ct_array.shape)
        patches = []
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
                ans, counter = update_ans(positions, preds, ans, counter)
                patches, positions = [], []

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
            ans, counter = update_ans(positions, preds, ans, counter)
            patches, positions = [], []

        if counter.min() == 0:
            raise Exception('Counter array has zero elements!')
        ans = ans / counter

        save_nii(ct, ans, ct_path)

        print('time:{:.3f} min'.format((time() - start) / 60), ct_path)

    print('test consuming time:', time() - start1)

