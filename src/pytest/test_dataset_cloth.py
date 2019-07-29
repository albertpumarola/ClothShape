
from src.options.config_parser import ConfigParser
from src.data.custom_dataset_data_loader import CustomDatasetDataLoader
from src.utils.tb_visualizer import TBVisualizer
from src.utils import util
from src.utils.plots import plot_estim
from collections import OrderedDict

# read options
config = ConfigParser().get_config()

# create data loader
train_data_loader = CustomDatasetDataLoader(config, is_for="train")
train_dataset = train_data_loader.load_data()
train_dataset_size = len(train_data_loader)
print('#training images = %d' % train_dataset_size)

tb_visualizer = TBVisualizer(config)

for i, data in enumerate(train_dataset):
    img = data['img']
    depth = data['depth']
    target = data['target']

    print("img", img.shape)
    print("depth", depth.shape)
    print("target", target.shape)

    visuals = OrderedDict()
    visuals['2_img'] = util.tensor2im(img.clone(), unnormalize=True)
    visuals['3_img'] = util.tensor2im(depth, unnormalize=True)
    visuals['1_img'] = plot_estim(util.tensor2im(img, to_numpy=True), target.numpy(), target.numpy())

    tb_visualizer.display_current_results(visuals, i, is_train=True)

    break