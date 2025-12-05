import os
from os.path import join
import time
from options.train_options import TrainOptions, TestOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

from fusion_dataset import Fusion_Testing_Dataset
from util import util
import os
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)

# Use CPU if CUDA is not available
if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU for inference (CUDA not available)")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    opt = TestOptions().parse()
    save_img_path = opt.results_img_dir
    if os.path.isdir(save_img_path) is False:
        print('Create path: {0}'.format(save_img_path))
        os.makedirs(save_img_path)
    opt.batch_size = 1
    dataset = Fusion_Testing_Dataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    dataset_size = len(dataset)
    print('#Testing images = %d' % dataset_size)

    model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')
    model.setup_to_test('coco_finetuned_mask_256_ffs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # Pure colorization: Use ONLY full-image model (netGComp) - NO bounding boxes, NO masks, NO fusion
        # This colorizes the entire image uniformly without any detector-based restrictions
        data_raw['full_img'] = data_raw['full_img'].to(device)
        full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_forward_without_box(full_img_data)
        model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))
    print('✅ Pure colorization complete - no bounding boxes or masks used')
