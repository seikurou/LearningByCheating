import time
import argparse


from pathlib import Path

import numpy as np
import torch
import tqdm

import glob
import os
import sys

import wandb

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass

import utils.bz_utils as bzu

from models.birdview import BirdViewPolicyModelSS
from utils.train_utils import one_hot
from utils.datasets.birdview_lmdb import get_birdview as load_data


# Maybe experiment with this eventually...
BACKBONE = 'resnet18'
GAP = 5
N_STEP = 5
SAVE_EPOCHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]

class LocationLoss(torch.nn.Module):
    def __init__(self, w=192, h=192, choice='l2'):
        super(LocationLoss, self).__init__()

        # IMPORTANT(bradyz): loss per sample.
        if choice == 'l1':
            self.loss = lambda a, b: torch.mean(torch.abs(a - b), dim=(1,2))
        elif choice == 'l2':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplemented("Unknown loss: %s"%choice)

        self.img_size = torch.FloatTensor([w,h]).cuda()

    def forward(self, pred_location, gt_location):
        '''
        Note that ground-truth location is [0,img_size]
        and pred_location is [-1,1]
        '''
        gt_location = gt_location / (0.5 * self.img_size) - 1.0

        return self.loss(pred_location, gt_location)


def _log_visuals(birdview, speed, command, loss, locations, _locations, size=16):
    import cv2
    import numpy as np
    import utils.carla_utils as cu

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = list()

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

        def _write(text, i, j):
            cv2.putText(
                    canvas, text, (cols[j], rows[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        def _dot(i, j, color, radius=2):
            x, y = int(j), int(i)
            canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color

        _command = {
                1: 'LEFT', 2: 'RIGHT',
                3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item()+1, '???')

        _dot(0, 0, WHITE)

        for x, y in locations[i]: _dot(x, y, BLUE)
        for x, y in (_locations[i] + 1) * (0.5 * 192): _dot(x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)

        images.append((loss[i].item(), canvas))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]


def train_or_eval(criterion, net, data, optim, is_train, config, is_first_epoch):
    if is_train:
        desc = 'Train'
        net.train()
    else:
        desc = 'Val'
        net.eval()

    if config['bev_freeze']:
        net.bev_net.eval()
        for p in net.bev_net.parameters():
            p.requires_grad = False

    total = 10 if is_first_epoch else len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)

    tick = time.time()

    for i, (birdview, location, command, speed, img) in iterator:
        birdview = birdview.to(config['device'])
        command = one_hot(command).to(config['device'])
        speed = speed.to(config['device'])
        location = location.float().to(config['device'])
        img = img.to(config['device'])

        pred_location, bev = net(birdview, speed, command, img, return_bev=True)
        if i % (len(data) // 10) == 0:
            bev_viz = bev.argmax(dim=1).detach().cpu().numpy()[0]
            bev_viz = colorize(bev_viz)
            wandb.log({
                'hiii': wandb.Image(bev_viz),
            }, commit=True)
        loss = criterion(pred_location, location)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        should_log = False
        should_log |= i % config['log_iterations'] == 0
        should_log |= not is_train
        should_log |= is_first_epoch

        if should_log:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = _log_visuals(
                    birdview, speed, command, loss,
                    location, pred_location)

            bzu.log.scalar(is_train=is_train, loss_mean=loss_mean.item())
            bzu.log.image(is_train=is_train, birdview=images)

        bzu.log.scalar(is_train=is_train, fps=1.0/(time.time() - tick))

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break


def train(config):
    bzu.log.init(config['log_dir'])
    name = os.path.basename(config['log_dir'])
    wandb.init(project='bevseg-lbc', name=name, sync_tensorboard=True, config=config)
    bzu.log.save_config(config)

    data_train, data_val = load_data(**config['data_args'], config=config)
    criterion = LocationLoss(w=192, h=192, choice='l1')
    net = BirdViewPolicyModelSS(**config['model_args'], config=config).to(config['device'])
    
    if config['resume']:
        log_dir = Path(config['log_dir'])
        checkpoints = list(log_dir.glob('model-*.th'))
        checkpoint = str(checkpoints[-1])
        print ("load %s"%checkpoint)
        net.load_state_dict(torch.load(checkpoint))
    
    optim = torch.optim.Adam(net.parameters(), lr=config['optimizer_args']['lr'])

    for epoch in tqdm.tqdm(range(config['max_epoch']+1), desc='Epoch'):
        train_or_eval(criterion, net, data_train, optim, True, config, epoch == 0)
        train_or_eval(criterion, net, data_val, None, False, config, epoch == 0)

        if epoch in SAVE_EPOCHS:
            torch.save(
                    net.state_dict(),
                    str(Path(config['log_dir']) / ('model-%d.th' % epoch)))

        bzu.log.end_epoch()

import torch
import numpy as np
from PIL import Image
import numpy as np
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import cv2

##############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    '''
    returns a numpy array representing a mapping: cmap[class][0] = r, cmap[class][1] = g, cmap[class][2] = b
    '''
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                         (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                         (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                         (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                         (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        '''
        gray_image: 1 x H x W (expected to be tensor with channel dim before w and h)
        '''
        # import pdb; pdb.set_trace()
        shape = gray_image.shape
        color_image = torch.ByteTensor(3, shape[1], shape[2]).fill_(0)

        for label in range(len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class ColorizeNumpy(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        '''
        gray_image: H x W
        '''
        shape = gray_image.shape
        color_image = np.zeros(shape=[shape[0], shape[1], 3], dtype=np.uint8)

        for label in range(len(self.cmap)):
            mask = (label == gray_image)
            color_image[mask, 0] = self.cmap[label][0]
            color_image[mask, 1] = self.cmap[label][1]
            color_image[mask, 2] = self.cmap[label][2]

        return color_image

color = ColorizeNumpy(19)
def colorize(gray_img):
    return color(gray_img)

CITYSCAPES_CLASSES = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]  # TrafficSigns
}

def as_cityscapes_palette(frame):
    """ Transforms the frame to the Carla cityscapes pallete.

    Note: this conversion is slow.
    """
    result = np.zeros((frame.shape[0], frame.shape[1], 3),
                      dtype=np.uint8)
    for key, value in CITYSCAPES_CLASSES.items():
        result[np.where(frame == key)] = value
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=1000)
    parser.add_argument('--max_epoch', default=1000)

    # Dataset.
    parser.add_argument('--dataset_dir', default='/raid0/dian/carla_0.9.6_data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--x_jitter', type=int, default=5)
    parser.add_argument('--y_jitter', type=int, default=0)
    parser.add_argument('--angle_jitter', type=int, default=5)
    parser.add_argument('--gap', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--cmd-biased', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--bev_net', type=str, default=None)
    parser.add_argument('--bev_freeze', action='store_true')
    parser.add_argument('--bev_channel', type=int, default=7)

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)

    parsed = parser.parse_args()

    config = {
            'log_dir': parsed.log_dir,
            'resume': parsed.resume,
            'log_iterations': parsed.log_iterations,
            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'optimizer_args': {'lr': parsed.lr},
            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                'n_step': N_STEP,
                'gap': GAP,
                'crop_x_jitter': parsed.x_jitter,
                'crop_y_jitter': parsed.y_jitter,
                'angle_jitter': parsed.angle_jitter,
                'max_frames': parsed.max_frames,
                'cmd_biased': parsed.cmd_biased,
                'num_workers': parsed.num_workers,
                },
            'model_args': {
                'model': 'birdview_dian',
                'input_channel': parsed.bev_channel,
                'backbone': BACKBONE,
                },
            'bev_net': parsed.bev_net,
            'bev_freeze': parsed.bev_freeze,
    }


    train(config)
