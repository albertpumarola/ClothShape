from src.options.config_parser import ConfigParser
from src.networks.networks import NetworksFactory
import torch

# get config
config = ConfigParser().get_config()

# get size
B = config["dataset"]["batch_size"]
S = config["dataset"]["image_size"]
n_img = config["dataset"]["img_nc"]
n_depth = config["dataset"]["depth_nc"]

# create network
nn_type = config["networks"]["reg"]["type"]
nn_hyper_params = config["networks"]["reg"]["hyper_params"]
nn = NetworksFactory.get_by_name(nn_type, **nn_hyper_params)

# run
img = torch.ones([B, n_img, S, S])
depth = torch.ones([B, n_depth, S, S])
y = nn(img, depth)
nn.print()
print(img.shape, y.shape)
