"""Default Parameter Values."""
from getpass import getuser

settings = {}

shared_folder = "/mnt/09434021-6a4d-4335-8aca-ce25516d62c7/SHARE/"
ozu_log_dir = (shared_folder + "tensorboard_log/zeroToAll")
log_dir = '/home/vvglab/tblogs' if getuser() == 'vvglab' else ozu_log_dir

settings['TENSORBOARD_LOGDIR'] = log_dir
settings['DATASETS_DIR'] = shared_folder+"Datasets/"
