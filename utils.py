import numba
import math
import numpy as np
import skimage.morphology
import time
from skimage.morphology import skeletonize_3d
import torch
import random


@numba.jit(nopython=True, fastmath=True)
def get_dist_map(patch, points, x, y, z):
    dist = np.zeros(patch.shape, dtype=np.float32)
    if len(points) == 0:
        return dist
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            for k in range(patch.shape[2]):
                d = find_nearest(points, x, y, z)
                dist[i, j, k] = d
    return dist


@numba.jit(nopython=True, fastmath=True)
def find_nearest(points, x, y, z):
    D = 999999
    for i in range(len(points)):
        p = points[i]
        Di = ((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) ** 0.5
        if Di < D:
            D = Di
    cut_off = 32
    if D > cut_off:
        d = 0.0
    else:
        d = math.exp(6 * (1 - D / cut_off)) - 1
        # d = 10*(1 - D / cut_off)
    return d


def get_coronary_LC(pred_array):
    pred_arg1 = get_large_single_component(pred_array.copy())
    pred_array[pred_arg1 == 1] = 0
    pred_arg2 = get_large_single_component(pred_array.copy())
    num_lc2 = pred_arg2.sum()

    pred_array[pred_arg2 == 1] = 0
    pred_arg3 = get_large_single_component(pred_array.copy())
    num_lc3 = pred_arg3.sum()
    if num_lc3 > num_lc2 / 2:
        pred_arg3[pred_arg1 == 1] = 1
        pred_arg3[pred_arg2 == 1] = 1
        pred_arg = pred_arg3
    else:
        pred_arg2[pred_arg1 == 1] = 1
        pred_arg = pred_arg2

    return pred_arg


def is_end(p, cl_array):
    r = 1
    x, y, z = p
    if cl_array[x - r:x + r + 1, y - r:y + r + 1, z - r:z + r + 1].sum() == 2:
        return True
    else:
        return False


def get_large_component(annotation):
    res = skimage.morphology.label(annotation)
    unique_labels, inverse, unique_counts = np.unique(res, return_inverse=True, return_counts=True)
    indics = unique_counts.argsort()[-5:][::-1]
    class_with_biggest_count = unique_labels[indics]
    annotation[
        (res != class_with_biggest_count[0]) &
        (res != class_with_biggest_count[1]) &
        (res != class_with_biggest_count[2]) &
        (res != class_with_biggest_count[3]) &
        (res != class_with_biggest_count[4])
        ] = 0

    return annotation


def get_large_single_component(annotation):
    res = skimage.morphology.label(annotation)
    unique_labels, inverse, unique_counts = np.unique(res, return_inverse=True, return_counts=True)
    indics = unique_counts[1:].argmax()
    class_with_biggest_count = unique_labels[1:][indics]
    annotation[res != class_with_biggest_count] = 0
    return annotation


def is_valid_patch(p, edge, gt_array):
    x, y, z = p
    overflow = x - edge < 0 or y - edge < 0 or z - edge < 0 or x + edge > gt_array.shape[0] or y + edge > \
               gt_array.shape[
                   1] or z + edge > gt_array.shape[2]
    if overflow:
        return False
    patch_gt = gt_array[x - edge:x + edge, y - edge:y + edge, z - edge:z + edge]
    return patch_gt.sum() > 20


def get_large_component2(annotation, labels=np.array([0, 1, 2, 3, 4])):
    closest_neighbour_map = annotation

    ans = np.zeros_like(closest_neighbour_map)
    for current_class_number in range(1, len(labels)):
        current_class_binary_mask = (closest_neighbour_map == current_class_number)

        res = skimage.morphology.label(current_class_binary_mask)

        unique_labels, inverse, unique_counts = np.unique(res, return_inverse=True, return_counts=True)

        not_background_classes = unique_labels[1:]
        not_background_classes_element_counts = unique_counts[1:]

        class_with_biggest_count = not_background_classes[not_background_classes_element_counts.argmax()]

        ans[res == class_with_biggest_count] = current_class_number

    return ans


@numba.jit(nopython=True, fastmath=True)
def get_full_field(points_mask, points_cl, output_line):
    output_full = np.zeros(output_line.shape)
    for i in range(len(points_mask)):
        x, y, z = points_mask[i]
        D = 999999
        D_p = None
        for j in range(len(points_cl)):
            p_c = points_cl[j]
            Di = ((x - p_c[0]) ** 2 + (y - p_c[1]) ** 2 + (z - p_c[2]) ** 2) ** 0.5
            if Di < D:
                D = Di
                D_p = p_c
        if D_p is not None:
            output_full[:, x, y, z] = output_line[:, D_p[0], D_p[1], D_p[2]]
        else:
            print('warning: D_p is None')
        if np.fabs(output_full[:, x, y, z]).max() == 0:
            print('warning: output line has zero vector')
    return output_full


def get_df_gt(mask, skip=4):
    import SimpleITK as sitk
    cl_mask = skeletonize_3d(mask.astype(np.int8))
    cl_mask[cl_mask == 2] = 1
    center_line_half_skip = np.argwhere(cl_mask)[::int(skip / 2)]
    output_line = np.zeros((3, mask.shape[0], mask.shape[1], mask.shape[2]))
    vector_length = 2
    for center in range(len(center_line_half_skip) - vector_length):
        p0 = center_line_half_skip[center]
        p2 = center_line_half_skip[center + vector_length]
        p1_dir = p2 - p0
        output_line[
        :,
        center_line_half_skip[center][0],
        center_line_half_skip[center][1],
        center_line_half_skip[center][2],
        ] = p1_dir / np.linalg.norm(p1_dir)

    points_mask = np.argwhere(mask == 1)
    output_full = get_full_field(points_mask, center_line_half_skip[:-vector_length], output_line)

    # visualize_vector(output_full, mask, cl_mask)

    # for test
    # output_mask = np.fabs(output_line).max(axis=0)
    # output_mask[output_mask > 0] = 1
    # img = sitk.GetImageFromArray(output_mask.astype(np.int8))
    # sitk.WriteImage(img, '../preds/output_mask_test.nii')
    #
    # img = sitk.GetImageFromArray(mask.astype(np.int8))
    # sitk.WriteImage(img, '../preds/output_mask_real.nii')
    # print('saved')

    return output_full


def get_point_from_vector(p, v, threshold):
    x, y, z = p[0], p[1], p[2]
    norm = (v ** 2).sum() ** 0.5
    norm_max = norm.max()
    if norm > norm_max * threshold:
        v = (v / norm)
        return np.ceil(np.array((x + v[0], y + v[1], z + v[2]))).astype(np.int8)
    return None


@numba.jit(nopython=True, fastmath=True)
def find_points_for_expand(ans, pred, points, sb, sx, sy, sz):
    for p in range(len(points)):
        b, x, y, z = points[p]
        if x == 0 or x == sx - 1 or y == 0 or y == sy - 1 or z == 0 or z == sz - 1:
            continue

        mark = True
        if pred[b, x + 1, y, z] and pred[b, x, y + 1, z] and pred[b, x, y, z + 1] and pred[b, x - 1, y, z] and pred[
            b, x, y - 1, z] and pred[b, x, y, z - 1]:
            mark = False
        if mark:
            ans[b, x, y, z] = 1
    return ans


# def get_norm_field(field_arr):
#     norm = (field_arr ** 2).sum(dim=1) ** 0.5
#     norm = norm.unsqueeze(dim=1)
#     return field_arr/norm


def get_mask_from_field(field_arr, mask):
    visual_predict(field_arr[-1], mask[-1])
    field_arr = (field_arr ** 2).sum(dim=1) ** 0.5
    threshold = field_arr.mean().item()
    field_arr[field_arr < threshold] = 0
    field_arr[field_arr >= threshold] = 1
    return field_arr


def visual_predict(field_arr, mask):
    import matplotlib.pyplot as plot

    # vec = field_arr.mean(dim=-1).mean(dim=-1).mean(dim=-1)
    # vec = vec / ((vec ** 2).sum() ** 0.5 + 1e-5) * 10

    img = np.zeros((field_arr.shape[-1], field_arr.shape[-2]))

    points = np.argwhere(mask == 1)
    print(len(points))
    for p in points:
        x, y, _ = p
        img[x, y] = 1

    cl_mask = skeletonize_3d(mask.astype(np.int8))
    cl_mask[cl_mask == 2] = 1

    points = np.argwhere(cl_mask == 1)
    for p in points:
        x, y, _ = p
        img[x, y] = 0.5

    x, y = (16, 16)

    plot.imshow(img, cmap='gray')

    for p in random.choices(np.argwhere(cl_mask == 1), k=10):
        x, y, _ = p
        vx, vy, _ = 2 * field_arr[:, p[0], p[1], p[2]]
        n = (vx ** 2 + vy ** 2) ** 0.5
        vx, vy = vx / n * 5, vy / n * 5
        plot.annotate("", xy=(y, x), xytext=(y + vy, x + vx), arrowprops=dict(arrowstyle="<-", color='g'))

    p = random.randint(1, 10)
    save_path = '../preds/vis_vector' + str(p) + '.png'
    plot.savefig(save_path)
    print('saved', save_path)
    plot.close()
    # exit(0)


def get_mask_from_field2(field_arr, pred):
    ans = np.zeros(pred.shape)
    ps = torch.nonzero(pred).cpu().detach().numpy()
    ans = find_points_for_expand(ans, pred.cpu().detach().numpy(), ps, *pred.shape)
    points = list(np.argwhere(ans == 1))

    # start = time.time()
    for i in range(2):
        l = len(points)
        for p in range(l):
            b, x, y, z = points[p]
            v = field_arr[b, :, x, y, z]

            next_p = get_point_from_vector(torch.tensor((x, y, z)), v)
            if next_p is not None and pred[b, next_p[0], next_p[1], next_p[2]] == 0:
                pred[b, next_p[0], next_p[1], next_p[2]] = 1
                points.append((b, next_p[0], next_p[1], next_p[2]))
    return pred


'''
    for i in range(field_arr_norm.shape[0]):
        norm_i = field_arr_norm[i]
        predi = pred[i]

        # calculate mean
        norm_i_cp = norm_i.clone()
        norm_i_cp[predi == 0] = 0
        num_non_zero = torch.nonzero(norm_i_cp).size(0)

        threshod = (norm_i_cp.sum() / num_non_zero) * 0.6

        field_arr_norm[i][norm_i > threshod] = 1
        field_arr_norm[i][norm_i <= threshod] = 0
    return field_arr_norm
'''


def visualize_vector(output_full, mask, cl_mask):
    import matplotlib.pyplot as plot
    import random

    img = np.zeros((output_full.shape[1], output_full.shape[2]))

    points = np.argwhere(mask == 1)
    for p in points:
        x, y, _ = p
        img[x, y] = 1

    points = np.argwhere(cl_mask == 1)
    for p in points:
        x, y, _ = p
        img[x, y] = 0.5

    plot.imshow(img, cmap='gray')

    for p in np.argwhere(cl_mask == 1):
        x, y, _ = p
        vx, vy, _ = 2 * output_full[:, p[0], p[1], p[2]]
        plot.annotate("", xy=(y, x), xytext=(y + vy, x + vx), arrowprops=dict(arrowstyle="<-", color='g'))

    for p in random.choices(np.argwhere(mask == 1), k=20):
        x, y, _ = p
        vx, vy, _ = 2 * output_full[:, p[0], p[1], p[2]]
        plot.annotate("", xy=(y, x), xytext=(y + vy, x + vx), arrowprops=dict(arrowstyle="<-", color='r'))

    p = random.randint(1, 10)
    save_path = '../preds/vis_vector' + str(p) + '.png'
    plot.savefig(save_path)
    plot.close()
    print('saved', save_path)


def update_hard_samples(lines):
    import pandas as pd
    df = pd.DataFrame(lines)
    df = df.sort_values(axis=0, by=1, ascending=False)
    lines = df.values.tolist()[:4]
    f = open('hard_samples.txt', 'w')
    f.write(str(lines))


def init_hard_samples():
    f = open('hard_samples.txt', 'w')
    f.write('[]')


def push_hard_sample_record(pos, info_record, loss_dice_unmean):
    for i in range(pos.shape[0]):
        id = pos[i].sum().item()
        mark = True
        for j in info_record:
            if j[0] == id:
                mark = False
        if mark:
            info_record.append([id, loss_dice_unmean[i].item(), pos[i, 0].item(), pos[i, 1].item(), pos[i, 2].item()])
    return info_record
