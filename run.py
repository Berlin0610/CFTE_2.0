import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import random
import numpy as np
from frames_dataset import FramesDataset
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.temporaldiscriminator import *
import torch
from train import *


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    
    seed_torch()
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
    parser = ArgumentParser()
    parser.add_argument("--config", default="./config/vox-256.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='./checkpoint/', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore") ###None
    parser.add_argument("--reload_epoch", default=None, help="quit the training and reload to train") ###None
    
    parser.add_argument("--gpu_num", default=4, help="GPU number")  #CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --device_ids 0,1,2,3
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    opt.device_ids = list(range(opt.gpu_num))
    
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, 'temporal_adaptive_3ref_512_new')

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'], **config['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        f_model = open('../generator.txt','w')
        print(generator, file=f_model)

    ######## single image  discriminator = spatial discriminator 
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'], **config['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)
        
        
    ######## multiple images  discriminator = temporal discriminator        
    temporaldiscriminator = TemporalMultiScaleDiscriminator(**config['model_params']['discriminator_params'], **config['common_params'])
    if torch.cuda.is_available():
        temporaldiscriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(temporaldiscriminator)        
        

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **config['common_params'])    
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_detector)
        

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)


    if opt.checkpoint is not None:
        print("Reload the checkpoint to Train...")
        retrain(config, generator, discriminator, temporaldiscriminator, kp_detector, opt.checkpoint,opt.reload_epoch, log_dir, opt.mode, opt.device_ids)
    else:
        print("Training...")
        train(config, generator, discriminator, temporaldiscriminator, kp_detector, opt.checkpoint, log_dir, opt.mode, opt.device_ids)
