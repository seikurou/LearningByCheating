import bird_view.utils.bz_utils as bzu
from bird_view.models import image
from pathlib import Path
import torch
import cv2

def get_image_agent(model_path):
    log_dir = Path(model_path).parent
    config = bzu.load_json(str(log_dir / 'config.json'))
    agent_args = config.get('agent_args', dict())
    model = image.ImagePolicyModelSS(**config['model_args'])
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    agent_args['model'] = model
    return image.ImageAgent(**agent_args)

observations = dict()
observations['rgb'] = cv2.imread('0561.png', cv2.IMREAD_UNCHANGED)[...,::-1]
observations['velocity'] = 0.
observations['command'] = 4
    # {
    #             1: 'LEFT', 2: 'RIGHT',
    #             3: 'STRAIGHT', 4: 'FOLLOW'}.get(4)

agent = get_image_agent("/data/efang/carla_lbc/ckpts/image/model-10.th")
controls = agent.run_step(observations)
print(controls)

# pip:
# loguru
# imageio
# tensorboardX