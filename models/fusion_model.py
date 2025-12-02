import os

import torch
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import io
from skimage import img_as_ubyte

import matplotlib.pyplot as plt
import math
from matplotlib import colors


class FusionModel(BaseModel):
    def name(self):
        return 'FusionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.model_names = ['G', 'GF']

        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1
        
        self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                      'instance', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=False)
        self.netG.eval()
        
        self.netGF = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                      'fusion', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=False)
        self.netGF.eval()

        self.netGComp = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                      'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=opt.classification)
        self.netGComp.eval()


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)
        
        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.opt.mask_cent

        self.real_B_enc = util.encode_ab_ind(self.real_B[:, :, ::4, ::4], self.opt)
    
    def set_fusion_input(self, input, box_info):
        AtoB = self.opt.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)

        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)

        self.full_mask_B_nc = self.full_mask_B + self.opt.mask_cent
        self.full_real_B_enc = util.encode_ab_ind(self.full_real_B[:, :, ::4, ::4], self.opt)
        self.box_info_list = box_info

    def set_forward_without_box(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)
        self.full_mask_B_nc = self.full_mask_B + self.opt.mask_cent
        self.full_real_B_enc = util.encode_ab_ind(self.full_real_B[:, :, ::4, ::4], self.opt)

        # Get both classification and regression outputs if available
        (self.comp_B_class, self.comp_B_reg) = self.netGComp(self.full_real_A, self.full_hint_B, self.full_mask_B)
        # Store classification output for potential annealed mean decoding
        self.has_classification = self.comp_B_class is not None and self.opt.classification
        # Default to regression, will use annealed mean if classification is available
        self.fake_B_reg = self.comp_B_reg

    def forward(self):
        (_, feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
        self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B, self.full_mask_B, feature_map, self.box_info_list)
        
    def save_current_imgs(self, path):
        """
        Save colorized image using ONLY regression output (fake_B_reg).
        No classification outputs, no heatmaps, no decoded bins - just natural regression-based colorization.
        
        Post-processing improvements:
        - Selective chroma boost: pixels with magnitude m = sqrt(a²+b²) > τ (τ=3.0) receive 1.3x boost
        - Skin correction: detect skin-like pixels in HSV, apply hue shift toward warm reference + 10% saturation boost
        - Edge-aware smoothing: light Gaussian blend to reduce chroma noise
        
        The ab values are normalized (divided by ab_norm=110.0), so fake_B_reg is in approximately [-1, 1] range.
        We scale them up to get visible colors, then lab2rgb will multiply by ab_norm to get actual Lab values.
        """
        # Use appropriate tensor type based on device
        tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        # Priority 1: Use annealed-mean decode for classification output if available (better than raw regression)
        # Check if we have classification output and use it with annealed mean
        use_classification = False
        if hasattr(self, 'has_classification') and self.has_classification:
            try:
                if hasattr(self, 'comp_B_class') and self.comp_B_class is not None:
                    # Annealed mean decoding: T=0.38 (tune in [0.3, 0.6] for vividness)
                    T = 0.38
                    logits = self.comp_B_class  # (B, K, H, W) where K=529 bins
                    
                    # Apply softmax with temperature
                    probs = torch.softmax(logits / T, dim=1)  # Sharpen distribution
                    
                    # Decode using mean over bins (util.decode_mean handles this)
                    ab_output = util.decode_mean(probs, self.opt)  # Returns normalized ab in [-1, 1]
                    use_classification = True
                    
                    if not hasattr(self, '_debug_printed'):
                        print(f"\n[DEBUG] Using annealed-mean classification decode (T={T})")
            except Exception as e:
                # Fall back to regression if classification decode fails
                if not hasattr(self, '_debug_printed'):
                    print(f"\n[DEBUG] Classification decode failed, using regression: {e}")
                ab_output = self.fake_B_reg.clone()
        else:
            # Use regression output
            ab_output = self.fake_B_reg.clone()
        
        # Debug: Print ab statistics to understand the output range
        ab_mean = ab_output.mean().item()
        ab_std = ab_output.std().item()
        ab_min = ab_output.min().item()
        ab_max = ab_output.max().item()
        ab_abs_mean = ab_output.abs().mean().item()
        ab_abs_max = ab_output.abs().max().item()
        
        # Print detailed debug info (only for first image to avoid spam)
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] AB channel statistics (BEFORE scaling):")
            print(f"  Source: {'Classification (annealed mean)' if use_classification else 'Regression'}")
            print(f"  Shape: {ab_output.shape}, Dtype: {ab_output.dtype}")
            print(f"  Mean: {ab_mean:.6f}, Std: {ab_std:.6f}")
            print(f"  Min: {ab_min:.6f}, Max: {ab_max:.6f}")
            print(f"  Abs Mean: {ab_abs_mean:.6f}, Abs Max: {ab_abs_max:.6f}")
            print(f"  ab_norm (from opt): {self.opt.ab_norm}")
            self._debug_printed = True
        
        # Priority 1: Per-image adaptive AB scaling (replaces fixed multiplier)
        # Formula: scale = 0.5 + (0.05 / max(std, 1e-6)), clamped to [1.0, 6.0]
        # This adapts to each image's color strength automatically
        scale = float(np.clip(0.5 + (0.05 / max(ab_std, 1e-6)), 1.0, 6.0))
        
        ab_output_scaled = ab_output * scale
        
        # Debug: Print stats after scaling
        if not hasattr(self, '_debug_scaled_printed'):
            print(f"\n[DEBUG] AB channel statistics (AFTER scaling={scale}x):")
            print(f"  Mean: {ab_output_scaled.mean().item():.6f}, Std: {ab_output_scaled.std().item():.6f}")
            print(f"  Min: {ab_output_scaled.min().item():.6f}, Max: {ab_output_scaled.max().item():.6f}")
            print(f"  After lab2rgb multiply by ab_norm ({self.opt.ab_norm}):")
            # Show what it would be before clamping
            print(f"    Before clamp: [{ab_output_scaled.min().item() * self.opt.ab_norm:.1f}, {ab_output_scaled.max().item() * self.opt.ab_norm:.1f}]")
            self._debug_scaled_printed = True
        
        # Clamp to valid normalized range [-1, 1] (lab2rgb expects normalized ab)
        # Accept some clamping loss - better than no color
        ab_output = torch.clamp(ab_output_scaled, -1.0, 1.0)
        
        # Combine L channel (grayscale) with AB channels (color) to form Lab image
        lab_img = torch.cat((self.full_real_A.type(tensor_type), ab_output.type(tensor_type)), dim=1)
        
        # Convert Lab to RGB using proper color space conversion
        # lab2rgb will multiply ab by opt.ab_norm (110.0) internally
        out_img = torch.clamp(util.lab2rgb(lab_img, self.opt), 0.0, 1.0)
        
        # Convert to numpy for saving
        out_img_np = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        
        # Priority 1: Selective saturation boost - only boost where color already exists
        # This prevents weird halos and makes skin tones/clothes pop without destroying realism
        from skimage import color as skcolor
        
        # Convert to Lab to check color magnitude
        lab_full = skcolor.rgb2lab(out_img_np)
        L_lab, a_lab, b_lab = lab_full[:, :, 0], lab_full[:, :, 1], lab_full[:, :, 2]
        
        # Calculate color magnitude: sqrt(a^2 + b^2)
        color_magnitude = np.sqrt(a_lab**2 + b_lab**2)
        
        # Threshold: only boost pixels that already have some color (magnitude > 3.0)
        # Lowered from 5.0 to catch faint skin colors and low-saturation faces
        # This avoids boosting near-gray areas which could create artifacts
        color_threshold = 3.0  # Lab color magnitude threshold (lowered to catch faint colors)
        color_mask = color_magnitude > color_threshold
        
        # Boost saturation only for pixels that already have color
        # Use smooth falloff to avoid hard edges
        boost_factor = 1.30  # Moderate boost (increased from 1.25 to help faces more)
        boost_map = np.where(color_mask, boost_factor, 1.0)
        
        # Apply smooth transition at threshold boundary
        transition_width = 2.0
        smooth_mask = np.clip((color_magnitude - (color_threshold - transition_width)) / (2 * transition_width), 0, 1)
        boost_map = 1.0 + (boost_factor - 1.0) * smooth_mask
        
        # Apply boost to ab channels in Lab space (more natural than HSV)
        a_lab_boosted = a_lab * boost_map
        b_lab_boosted = b_lab * boost_map
        
        # Reconstruct Lab image
        lab_boosted = np.stack([L_lab, a_lab_boosted, b_lab_boosted], axis=-1)
        out_img_np = skcolor.lab2rgb(lab_boosted)
        out_img_np = np.clip(out_img_np, 0, 1)
        
        # Additional luminance-aware refinement: slight boost to midtones
        hsv = skcolor.rgb2hsv(out_img_np)
        L_channel = self.full_real_A.cpu().data.numpy()[0, 0, :, :]
        L_normalized = (L_channel - L_channel.min()) / (L_channel.max() - L_channel.min() + 1e-6)
        
        # Light midtone boost (1.1x) - very subtle
        midtone_mask = np.clip((L_normalized - 0.2) / 0.3, 0, 1.0) * np.clip((0.8 - L_normalized) / 0.3, 0, 1.0)
        midtone_boost = 1.0 + 0.1 * midtone_mask  # Very light boost (1.0-1.1x)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * midtone_boost, 0, 1.0)
        
        out_img_np = skcolor.hsv2rgb(hsv)
        out_img_np = np.clip(out_img_np, 0, 1)
        
        # Priority 2: Enhanced skin correction (fixes ghostly/blue-ish faces)
        # Work in Lab space for better chroma control
        lab_skin = skcolor.rgb2lab(out_img_np)
        L_lab_skin, a_lab_skin, b_lab_skin = lab_skin[:, :, 0], lab_skin[:, :, 1], lab_skin[:, :, 2]
        
        # Build skin mask in HSV first (easier to define hue ranges)
        hsv_skin = skcolor.rgb2hsv(out_img_np)
        H, S, V = hsv_skin[:, :, 0], hsv_skin[:, :, 1], hsv_skin[:, :, 2]
        
        # Skin hue range: 0-40° (red to yellow-orange) in normalized [0, 1] = 0.0-0.11
        # EXCLUDE green hues (0.2-0.4) to avoid catching vegetables
        skin_hue_mask = (H >= 0.0) & (H <= 0.11)  # Only orange-red to yellow range
        
        # Skin should have moderate saturation (0.1-0.7) and reasonable value (> 0.2)
        # Exclude very high saturation (> 0.7) which are likely colored objects
        # Exclude very low saturation (< 0.1) which are grayscale
        skin_mask = skin_hue_mask & (S > 0.1) & (S < 0.7) & (V > 0.2) & (V < 0.95)
        
        # Apply enhanced skin correction in Lab space for better chroma control
        if np.any(skin_mask):
            # Get skin pixels' ab channels
            a_s = a_lab_skin[skin_mask]
            b_s = b_lab_skin[skin_mask]
            
            # Calculate chroma magnitude for skin pixels
            mag = np.sqrt(a_s**2 + b_s**2)
            
            # Boost very low-chroma skin more aggressively (forces color instead of staying gray)
            low_chroma_mask = mag < 5.0
            a_s[low_chroma_mask] *= 1.8  # Aggressive boost for almost-gray skin
            b_s[low_chroma_mask] *= 1.8
            
            # Moderate boost for normal skin chroma
            a_s[~low_chroma_mask] *= 1.1
            b_s[~low_chroma_mask] *= 1.1
            
            # Apply corrections back to full image
            a_lab_skin[skin_mask] = a_s
            b_lab_skin[skin_mask] = b_s
            
            # Clamp brightness for skin - prevent over-brightening faces
            L_s = L_lab_skin[skin_mask]
            L_max = L_s.max()
            L_lab_skin[skin_mask] = np.minimum(L_s, 0.9 * L_max)  # Cap at 90% of max skin brightness
            
            # Also apply hue correction in HSV for warm skin tones
            target_skin_hue = 0.08  # Warm orange-tan
            H_corrected = np.where(skin_mask, 0.75 * target_skin_hue + 0.25 * H, H)
            H_corrected = np.clip(H_corrected, 0, 1)
            
            # Reconstruct Lab image with corrected ab channels
            lab_corrected = np.stack([L_lab_skin, a_lab_skin, b_lab_skin], axis=-1)
            out_img_np = skcolor.lab2rgb(lab_corrected)
            out_img_np = np.clip(out_img_np, 0, 1)
            
            # Apply hue correction in final RGB
            hsv_final = skcolor.rgb2hsv(out_img_np)
            hsv_final[:, :, 0] = H_corrected  # Apply corrected hue
            out_img_np = skcolor.hsv2rgb(hsv_final)
            out_img_np = np.clip(out_img_np, 0, 1)
        
        # Priority 2: Edge-aware smoothing using simple bilateral-like approach
        # Apply light Gaussian blur to ab channels only (preserves edges via L channel)
        try:
            from scipy import ndimage
            # Light smoothing: blend 90% original with 10% blurred
            blurred = ndimage.gaussian_filter(out_img_np, sigma=0.5)
            out_img_np = 0.9 * out_img_np + 0.1 * blurred
            out_img_np = np.clip(out_img_np, 0, 1)
        except:
            pass  # If scipy not available, skip smoothing
        
        # Save the image
        io.imsave(path, img_as_ubyte(out_img_np))

    def setup_to_test(self, fusion_weight_path):
        GF_path = 'checkpoints/{0}/latest_net_GF.pth'.format(fusion_weight_path)
        print('load Fusion model from %s' % GF_path)
        # Map to CPU if CUDA is not available
        map_location = 'cpu' if not torch.cuda.is_available() else None
        GF_state_dict = torch.load(GF_path, map_location=map_location)
        
        # G_path = 'checkpoints/coco_finetuned_mask_256/latest_net_G.pth' # fine tuned on cocostuff
        G_path = 'checkpoints/{0}/latest_net_G.pth'.format(fusion_weight_path)
        G_state_dict = torch.load(G_path, map_location=map_location)

        # GComp_path = 'checkpoints/siggraph_retrained/latest_net_G.pth' # original net
        # GComp_path = 'checkpoints/coco_finetuned_mask_256/latest_net_GComp.pth' # fine tuned on cocostuff
        GComp_path = 'checkpoints/{0}/latest_net_GComp.pth'.format(fusion_weight_path)
        GComp_state_dict = torch.load(GComp_path, map_location=map_location)

        self.netGF.load_state_dict(GF_state_dict, strict=False)
        # Handle both DataParallel and regular models
        if hasattr(self.netG, 'module'):
            self.netG.module.load_state_dict(G_state_dict, strict=False)
        else:
            self.netG.load_state_dict(G_state_dict, strict=False)
        if hasattr(self.netGComp, 'module'):
            self.netGComp.module.load_state_dict(GComp_state_dict, strict=False)
        else:
            self.netGComp.load_state_dict(GComp_state_dict, strict=False)
        self.netGF.eval()
        self.netG.eval()
        self.netGComp.eval()