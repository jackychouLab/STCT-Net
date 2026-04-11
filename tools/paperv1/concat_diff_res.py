import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

seq_list = [2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77]
title_list = [
    'image',
    'ra',
    'confmap',
    'cdc',
    'wo_PCA',
    'wo_STCT',
    'wo_backbone',
    'wo_softBCE',
    'full',
]
image_root = '/mnt/d/paperv1/important/image'
ra_root = '/mnt/d/paperv1/important/ra'
gt_root = '/mnt/d/paperv1/important/confmap'
cdc_root = '/mnt/d/paperv1/important/cdc'
wo_pca_root = '/mnt/d/paperv1/important/wo_PCA'
wo_stct_root = '/mnt/d/paperv1/important/wo_STCT'
wo_fe_root = '/mnt/d/paperv1/important/wo_backbone'
wo_loss_root = '/mnt/d/paperv1/important/wo_softBCE'
full_root = '/mnt/d/paperv1/important/full'

save_root = '/mnt/d/paperv1/important/concat'
if not os.path.exists(save_root):
    os.mkdir(save_root)

for seq_id in seq_list:
    for frame_id in range(0, 300):
        image_path = os.path.join(image_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        ra_path = os.path.join(ra_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        gt_path = os.path.join(gt_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        cdc_path = os.path.join(cdc_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        wo_pca_path = os.path.join(wo_pca_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        wo_stct_path = os.path.join(wo_stct_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        wo_fe_path = os.path.join(wo_fe_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        wo_loss_path = os.path.join(wo_loss_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")
        full_path = os.path.join(full_root, f"uav_seqs_{seq_id}", f"{frame_id:09d}.jpg")

        images = []
        images.append(mpimg.imread(image_path))
        images.append(mpimg.imread(ra_path))
        images.append(mpimg.imread(gt_path))
        images.append(mpimg.imread(cdc_path))
        images.append(mpimg.imread(wo_pca_path))
        images.append(mpimg.imread(wo_stct_path))
        images.append(mpimg.imread(wo_fe_path))
        images.append(mpimg.imread(wo_loss_path))
        images.append(mpimg.imread(full_path))
        plt.close('all')
        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        for ax, img, title in zip(axes, images, title_list):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{title}")
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        sub_save_path = os.path.join(save_root, f"{seq_id:02d}-{frame_id:09d}.jpg")
        plt.savefig(sub_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)