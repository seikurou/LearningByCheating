import cv2
import numpy as np

import torch
import torch.nn as nn

from . import common
from .agent import Agent
from .controller import PIDController, CustomController
from .controller import ls_circle


STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5


def regression_base():
    return nn.Sequential(
            nn.ConvTranspose2d(640,256,4,2,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,0),
            nn.BatchNorm2d(64),
            nn.ReLU(True))


def spatial_softmax_base():
    return nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ConvTranspose2d(640,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True))


class BirdViewPolicyModelSS(common.ResnetBase):
    def __init__(self, backbone='resnet18', input_channel=7, n_step=5, all_branch=False, **kwargs):
        super().__init__(backbone=backbone, input_channel=input_channel, bias_first=False)

        self.deconv = spatial_softmax_base()
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(48,48,STEPS)) for i in range(COMMANDS)
        ])
        
        self.all_branch = all_branch

        self.config = kwargs['config']
        bev_name = self.config['bev_net']
        if bev_name == 'vpn':
            pretrained = '/data/ck/BESEG/baseline_vpn/runs/lbc_noise_7/_best.pth.tar'
            from vpn.train_carla import parse_args_and_construct_model
            vpn_args = ''' --fc-dim 256 --use-mask false --transform-type fc --input-resolution 512 --label-resolution 32 --n-views 1 --num-class {bev_channel}  \
            --use_depth False \
            --use_segmented False \
            --SegSize 192 \
            --dataset bevseg \
            --multiclass_binary_entropy False \
            --resume {pretrained} \
            '''.format(
                bev_channel=input_channel,
                pretrained=pretrained
            )
            self.bev_net = parse_args_and_construct_model(vpn_args)

    def forward(self, bird_view, velocity, command, img=None, return_bev=False):

        if self.config['bev_net'] and self.config['bev_net'] != 'None':
            assert type(img) != type(None), 'when using bev_net, we need the input img to generate bev'
            bev = self.bev_net(img)
            bev = nn.functional.interpolate(bev, size=192, mode='bilinear', align_corners=False)
            bird_view = bev

        h = self.conv(bird_view)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[...,None,None,None].repeat((1,128,kh,kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)
            
        location_pred = common.select_branch(location_preds, command)

        ret = [location_pred]
        if self.all_branch:
            ret.append(location_preds)
        if return_bev:
            ret.append(bird_view)
        return ret


class BirdViewAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5, **kwargs):
        super().__init__(**kwargs)

        self.speed_control = PIDController(K_P=1.0, K_I=0.1, K_D=2.5)

        if steer_points is None:
            steer_points = {"1": 3, "2": 2, "3": 2, "4": 2}
        
        if pid is None:
            pid = {
                "1" : {"Kp": 1.0, "Ki": 0.1, "Kd":0}, # Left
                "2" : {"Kp": 1.0, "Ki": 0.1, "Kd":0}, # Right
                "3" : {"Kp": 0.8, "Ki": 0.1, "Kd":0}, # Straight
                "4" : {"Kp": 0.8, "Ki": 0.1, "Kd":0}, # Follow
            }
            
        self.turn_control = CustomController(pid)
        self.steer_points = steer_points

        self.gap = gap

    def run_step(self, observations, teaching=False):
        birdview = common.crop_birdview(observations['birdview'], dx=-10)
        speed = np.linalg.norm(observations['velocity'])
        command = self.one_hot[int(observations['command']) - 1]
        img = np.uint8(observations['rgb'])
        from bird_view.utils.datasets.birdview_lmdb import preprocess_img
        img = preprocess_img(img)
        img = torch.FloatTensor([img]).to(self.device)


        with torch.no_grad():
            _birdview = self.transform(birdview).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)

            if self.model.all_branch:
                _locations, _ = self.model(_birdview, _speed, _command)
            else:
                _locations, bev = self.model(_birdview, _speed, _command, img=img, return_bev=True)
            _locations = _locations.squeeze().detach().cpu().numpy()
    
        _map_locations = _locations
        # Pixel coordinates.
        _locations = (_locations + 1) / 2 * CROP_SIZE

        targets = list()

        for i in range(STEPS):
            pixel_dx, pixel_dy = _locations[i]
            pixel_dx = pixel_dx - CROP_SIZE / 2
            pixel_dy = CROP_SIZE - pixel_dy

            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy]) / PIXELS_PER_METER

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        target_speed = 0.0

        for i in range(1, SPEED_STEPS):
            pixel_dx, pixel_dy = _locations[i]
            prev_dx, prev_dy = _locations[i-1]

            dx = pixel_dx - prev_dx
            dy = pixel_dy - prev_dy
            delta = np.linalg.norm([dx, dy])

            target_speed += delta / (PIXELS_PER_METER * self.gap * DT) / (SPEED_STEPS-1)

        _cmd = int(observations['command'])
        n = self.steer_points.get(str(_cmd), 1)
        targets = np.concatenate([[[0, 0]], targets], 0)
        c, r = ls_circle(targets)
        closest = common.project_point_to_circle(targets[n], c, r)

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)
        steer = self.turn_control.run_step(alpha, _cmd)
        throttle = self.speed_control.step(target_speed - speed)
        brake = 0.0

        if target_speed < 1.0:
            steer = 0.0
            throttle = 0.0
            brake = 1.0

        self.debug['locations_birdview'] = _locations[:,::-1].astype(int)
        self.debug['target'] = closest
        self.debug['target_speed'] = target_speed
        self.debug['bev'] = bev

        control = self.postprocess(steer, throttle, brake)
        if teaching:
            return control, _map_locations
        else:
            return control
