# import the necessary packages
import torch
# set device to 'cuda' if CUDA is available, 'mps' if MPS is available,
# or 'cpu' otherwise for model training and testing
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# define model hyperparameters
LR = 0.005
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 256
EMBEDDING_DIM = 512
EPOCHS = 200
KLD_WEIGHT = 0.00025
# define the dataset path
# DATASET_PATH = r"F:\pycharm_projects_personal\smile_not_smile\VAE\data\Celeba/img_align_celeba"
# DATASET_PATH = r"F:\pycharm_projects_personal\smile_not_smile\VAE\data\Celeba\kaggle\train"
DATASET_PATH = r"A:\pycharm_projects\kaggle_smile_no_smile\datasets\from_kaggle\test_all"
DATASET_ATTRS_PATH = r"F:\pycharm_projects_personal\smile_not_smile\VAE\data\Celeba/list_attr_celeba.csv"
# parameters for morphing process
NUM_FRAMES = 50
FPS = 5
# list of labels for image attribute modifier experiment
LABELS = ["Eyeglasses", "Smiling", "Attractive", "Male", "Blond_Hair"]