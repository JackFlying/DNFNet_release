import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize(image_root, probe_img, probe_roi, color, save_name):
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
    save_root_dir = "/home/linhuadong/DNFNet/tools/system/cache"
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    plt.savefig(os.path.join(save_root_dir, save_name + '.png'), bbox_inches='tight')

def vis_search_result(results):
    colors = ['red', 'green', 'yellow']
    visualize(results['image_root'], results['probe_img'], results['probe_roi'], colors[2], 'query') # query
    for j in range(5):
        info = results['gallery'][j]
        visualize(results['image_root'], info['img'], info['roi'][:4], colors[info['correct']], save_name=f'top-{j+1}')  # gallery

