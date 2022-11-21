import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
import torch

baseline_root = '/home/linhuadong/CGPS/jobs/prw_baseline/evaluate_result.pkl'
root = '/home/linhuadong/CGPS/jobs/prw_baseline/evaluate_result.pkl'
dir = '/home/linhuadong/CGPS/jobs/prw_baseline/evaluate_vis/'
dir_cat = '/home/linhuadong/CGPS/jobs/prw_baseline/evaluate_vis_cat/'
with open(baseline_root, "rb") as file:
    evaluate_result_baseline = pickle.load(file)

with open(root, "rb") as file:
    evaluate_result = pickle.load(file)

image_root = evaluate_result['image_root']
results_baseline = evaluate_result_baseline['results']
results = evaluate_result['results']

def drap_map(feature_maps, imgs, folder, save_name):
    """
        :feature_maps: [2048, 7(H), 7(W)]
    """
    feature_maps = torch.tensor(feature_maps)
    feature_maps = feature_maps.unsqueeze(0)
    outputs = (feature_maps ** 2).sum(1)
    b, h, w = outputs.shape
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)

    # imgs = torch.tensor(imgs).unsqueeze(0)
    imgs = transforms.ToTensor()(imgs).unsqueeze(0)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    GRID_SPACING = 5
        
    for j in range(outputs.size(0)):
        # RGB image
        img = imgs[j, ...]
        for t, m, s in zip(img, imagenet_mean, imagenet_std):
            t.mul_(s).add_(m).clamp_(0, 1)
        img_np = np.uint8(np.floor(img.numpy() * 255))
        img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)
        height, width = img_np.shape[0], img_np.shape[1]

        # activation map
        am = outputs[j, ...].numpy()  # import matplotlib.pyplot as plt;plt.imshow(grid_img);plt.show()
        am = cv2.resize(am, (width, height))
        am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))  # import scipy.io as sio; sio.savemat('np_vector.mat', {'vect':am})
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

        # overlapped
        img_ratio = 0.7
        overlapped = img_np * img_ratio + am * (1 - img_ratio)
        overlapped[overlapped > 255] = 255
        overlapped = overlapped.astype(np.uint8)

        # save images in a single figure (add white spacing between images)
        # from left to right: original image, activation map, overlapped image
        grid_img = 255 * np.ones((height, 2 * width + 1 * GRID_SPACING, 3), dtype=np.uint8)
        grid_img[:, :width, :] = img_np[:, :, ::-1]
        grid_img[:, 1 * width + 1 * GRID_SPACING:, :] = overlapped
        if not os.path.exists(folder):
            os.makedirs(dir)
        cv2.imwrite(os.path.join(folder, save_name), grid_img)

def visualize(probe_img, probe_roi, idx, color, name, k, roi_feats):
    folder = os.path.join(dir, str(idx))
    if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder)

    file_root = os.path.join(image_root, probe_img)
    img = Image.open(file_root)
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    ax = plt.gca()
    down_left_x, down_left_y = probe_roi[0], probe_roi[1]
    width = probe_roi[2] - probe_roi[0]
    height = probe_roi[3] - probe_roi[1]
    rect = plt.Rectangle((down_left_x, down_left_y), width, height, fill=False, edgecolor=color, linewidth=1)
    ax.add_patch(rect)

    # 前两个坐标点是左上角坐标, 后两个坐标点是右下角坐标, width在前, height在后
    crop_img = img.crop((down_left_x, down_left_y, down_left_x + width, down_left_y + height))
    if name == "query":
        save_name = "{}.jpg".format(name)
        save_name_map = "map_{}.jpg".format(name)
    else:
        save_name = "{}_top_{}.jpg".format(name, k)
        save_name_map = "map_{}_top_{}.jpg".format(name, k)
        drap_map(roi_feats, crop_img, folder, save_name_map)
    plt.savefig(os.path.join(folder, save_name), dpi=200, bbox_inches='tight')

def search(result_baseline, result, idx):
    # if result_baseline['acc'][0] >= result['acc'][0]:
    #     return False
    colors = ['red', 'green']
    # print("baseline: ap:{}, acc:{}".format(result_baseline['ap'], result_baseline['acc']))
    # print("this: ap:{}, acc:{}".format(result['ap'], result['acc']))
    visualize(result['probe_img'], result['probe_roi'], idx, 'yellow', 'query', 0, None) # probe
    for j in range(0, 5, 1):
        baselin_info = result_baseline['gallery'][j]
        visualize(baselin_info['img'], baselin_info['roi'][:4], idx, colors[baselin_info['correct']], "baseline", j + 1, baselin_info['roi_feats'])  # gallery

        info = result['gallery'][j]
        visualize(info['img'], info['roi'][:4], idx, colors[info['correct']], "this", j + 1, info['roi_feats'])  # gallery
    return True

def cat_image(cdir, name):
    if not os.path.exists(dir_cat):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dir_cat)
    
    query = Image.open(os.path.join(cdir, "query.jpg"))
    width, height = query.size
    block = Image.new('RGB', (width // 3, height), (255, 255, 255))
    cat_base, cat_this, cat_base_map, cat_this_map = query, query, block, block
    for i in range(5):
        base_img = Image.open(os.path.join(cdir, "baseline_top_{}.jpg".format(i + 1)))
        cat_base = np.concatenate([cat_base, base_img], axis=0)

        this_img = Image.open(os.path.join(cdir, "this_top_{}.jpg".format(i + 1)))
        cat_this = np.concatenate([cat_this, this_img], axis=0)

        map_base_img = Image.open(os.path.join(cdir, "map_this_top_{}.jpg".format(i + 1)))
        map_base_img = map_base_img.resize((width // 3, height))
        cat_base_map = np.concatenate([cat_base_map, map_base_img], axis=0)

        map_this_img = Image.open(os.path.join(cdir, "map_this_top_{}.jpg".format(i + 1)))
        map_this_img = map_this_img.resize((width // 3, height))
        cat_this_map = np.concatenate([cat_this_map, map_this_img], axis=0)

    cat_img = np.concatenate([cat_base_map, cat_base, cat_this, cat_this_map], axis=1)
    plt.figure(figsize=(15, 15))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cat_img)
    plt.savefig(dir_cat + "/{}.jpg".format(name), dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    count = 0
    for idx in range(0, len(results_baseline), 1):
        if idx > 100:
            break
        if search(results_baseline[idx], results[idx], idx):
            count += 1
    print(count, count / len(results_baseline))

    filedir = os.listdir(dir)
    for i, name in enumerate(filedir):
        cdir = os.path.join(dir, name)
        imgfile = os.listdir(cdir)
        cat_image(cdir, name)