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
from color_editor import interactive_edit
from PIL import Image
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
    
    # Check if interactive editing is enabled (via command line or prompt)
    enable_editing = opt.enable_editing
    if not enable_editing:
        try:
            response = input("\n🔧 Enable interactive post-editing? [y/n, default=n]: ").strip().lower()
            enable_editing = response == 'y'
        except (EOFError, KeyboardInterrupt):
            # If running non-interactively, default to False
            enable_editing = False
            print("\n⚠️  Running in non-interactive mode. Post-editing disabled.")
    
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # Pure colorization: Use ONLY full-image model (netGComp) - NO bounding boxes, NO masks, NO fusion
        # This colorizes the entire image uniformly without any detector-based restrictions
        data_raw['full_img'] = data_raw['full_img'].to(device)
        full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_forward_without_box(full_img_data)
        
        # Save the initial colorized image
        out_path = join(save_img_path, data_raw['file_id'][0] + '.png')
        model.save_current_imgs(out_path)
        
        # Optional post-editing stage
        if enable_editing:
            # Load the saved image for editing
            img = np.array(Image.open(out_path).convert("RGB"))
            
            # Interactive editing
            edited_img = interactive_edit(img, out_path)
            
            # Save the edited version
            Image.fromarray(edited_img).save(out_path)
            print(f"💾 Saved edited image to {out_path}\n")
    
    print('✅ Pure colorization complete - no bounding boxes or masks used')
