"""
训练脚本
"""

import os
from time import time
import numpy as np
import utils
import SimpleITK as sitk
# from config import root
import glob
import torch
edge = 16

from net.ResUnet import net
from skimage.morphology import skeletonize_3d


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
center_p = edge_vec = np.array((edge, edge, edge), dtype=np.int)


def update_ans(pred_patch, ans, counter, p):
    x, y, z = p
    ans[:, x - edge:x + edge, y - edge:y + edge, z - edge:z + edge] += pred_patch
    counter[x - edge:x + edge, y - edge:y + edge, z - edge:z + edge] += 1
    pred_arg = torch.argmax(pred_patch, dim=0).cpu().detach().numpy()
    return ans, counter, pred_arg


def smooth_visited(visited, p):
    x, y, z = p
    e = edge // 2
    return visited[x - e:x + e, y - e:y + e, z - e:z + e].sum() >= 1


def get_new_points(pred_arg, p):
    # for calculate world coordinates
    anchor_p = p - edge_vec

    #  get center line for the patch
    cl_array = skeletonize_3d(pred_arg)

    # remove near points
    e = edge - 2
    cl_array[center_p[0] - e:center_p[0] + e, center_p[1] - e:center_p[1] + e, center_p[2] - e: center_p[2] + e] = 0
    points = np.argwhere(cl_array > 0)

    # return world coordinates
    return anchor_p + points


def do_segment(ct_array, fr_array, tp_array, net, ht_array):
    # result
    ans = np.zeros((2,) + ct_array.shape)
    ans[1, tp_array == 1] = 1
    ans = torch.tensor(ans).float()
    counter = torch.zeros(ct_array.shape)

    # for bfs
    visited = np.zeros(ct_array.shape)
    queue = np.zeros((100000, 3)).astype(int)
    head = tail = 0

    # search seed
    cl_array = skeletonize_3d(tp_array)
    cl = np.argwhere(cl_array > 0)
    for p in cl:
        if not smooth_visited(visited, p) and ht_array[p[0], p[1], p[2]] == 1:
            queue[tail, :] = p
            tail += 1
            visited[p[0], p[1], p[2]] = 1

    # do bfs
    while head < tail:
        # pop anchor point from queue
        x, y, z = next_p = queue[head, :]
        head += 1

        if not utils.is_valid_patch(next_p, edge, ct_array):
            continue

        # crop patch at anchor point
        patch = ct_array[x - edge:x + edge, y - edge:y + edge, z - edge:z + edge]
        patch_fr = fr_array[x - edge:x + edge, y - edge:y + edge, z - edge:z + edge]
        input_patch = torch.tensor(np.array([patch, patch_fr])).cuda().float().unsqueeze(dim=0)

        # test patch which is at anchor point
        with torch.no_grad():
            pred_patch = net(input_patch).squeeze(dim=0).cpu().detach()

        # update to whole result array
        ans, counter, pred_arg = update_ans(pred_patch, ans, counter, next_p)

        # update center line and find new points
        for p in get_new_points(pred_arg, next_p):
            if not smooth_visited(visited, p) and ht_array[p[0], p[1], p[2]] == 1:
                queue[tail, :] = p
                tail += 1
                visited[p[0], p[1], p[2]] = 1
    ans[0, counter == 0] = 1
    ans_arg = torch.argmax(ans, dim=0).cpu().detach().numpy()

    return ans_arg


def save_nii(hander, pred_array, ct_path):
    nii = sitk.GetImageFromArray(pred_array.astype(np.int8))
    nii.SetDirection(hander.GetDirection())
    nii.SetOrigin(hander.GetOrigin())
    nii.SetSpacing(hander.GetSpacing())
    save_path = ct_path.replace('CT', 'CR')
    sitk.WriteImage(nii, save_path)


def main(net, file_list):
    start1 = time()
    for index in range(len(file_list)):
        start = time()
        ct_path = file_list[index].replace('\n', '')
        name = ct_path.split('/')[-2]
        fr_path = ct_path.replace('CT', 'FR')
        tp_path = ct_path.replace('CT', 'TP')
        ht_path = ct_path.replace('CT', 'HT')
        save_path = ct_path.replace('CT', 'CR')
        # if os.path.exists(save_path):
        #     print('existed skip...')
        #     continue

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path)
        fr = sitk.ReadImage(fr_path)
        tp = sitk.ReadImage(tp_path)
        ht = sitk.ReadImage(ht_path)
        ct_array = sitk.GetArrayFromImage(ct)
        fr_array = sitk.GetArrayFromImage(fr)
        tp_array = sitk.GetArrayFromImage(tp)
        ht_array = sitk.GetArrayFromImage(ht)

        pred_array = do_segment(ct_array, fr_array, tp_array, net, ht_array)
        pred_array = utils.get_coronary_LC(pred_array)


        # for visualization
        save_nii(ct, pred_array, ct_path)

        print('time:{:.3f} min'.format((time() - start) / 60), ct_path)

    print('test consuming time:', time() - start1)


if __name__ == '__main__':
    pretrained_model_path = './module/best_ct2mask.pth'
    file_list = glob.glob('./RotterdamCoronaryDataset/*/CT.nii.gz')
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(pretrained_model_path), strict=False)
    main(net, file_list)

