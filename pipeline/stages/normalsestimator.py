import numpy as np
import cv2
import torch
from torchvision import transforms

from pipeline.surface_normals.NNET import NNET
from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.logging import log_image, im_logging_enabled

__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

class normals_config:
    architecture="BN"
    sampling_ratio=0.4
    importance_ratio=0.7

# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out

class PipelineNormalsEstimator(PipelineStep):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        
        self.device = torch.device('cuda:0')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        
        # parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
        # parser.add_argument("--pretrained", required=True, type=str, help="{nyu, scannet}")
        # parser.add_argument('--sampling_ratio', type=float, default=0.4)
        # parser.add_argument('--importance_ratio', type=float, default=0.7)

        #python test.py --pretrained scannet --architecture BN
        
        self.model = NNET(normals_config).to(self.device)
        self.model = load_checkpoint(self.pipeline.config.normals_model_path, self.model)
        self.model.eval()

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.NormalsEstimator

    @property
    def required_keys(self) -> list:
        return ["downscaled"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        print("Normals")
        
        #img = cv2.resize(data["image"], (640, 480))
    

        # if "downscaled" not in data:
        #     output = data["semantic_probs"]
        #     h, w = output[0].shape
        #     shape = (w, h)
        #     if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
        #         data["downscaled"] = cv2.resize(data["image"], shape)
        #     else:
        #         data["downscaled"] = data["image"]

        # img = data["downscaled"]

        img = cv2.resize(data["image"], (1024, 768))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)

        img = img.unsqueeze_(0)
        img = img.to(self.device)

        with torch.no_grad():
            norm_out_list, _, _ = self.model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # 2. predicted normal
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb[0].astype(np.uint8)                  # (H, W, 3)

            data["surface_normals"] = pred_norm_rgb

        # norm_out_list, _, _ = self.model(img)
        # norm_out = norm_out_list[-1]

        
 

