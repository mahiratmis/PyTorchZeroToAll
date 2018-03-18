"""Default Parameter Values."""
from getpass import getuser

settings = {}
ozu_log_dir = ("/mnt/09434021-6a4d-4335-8aca-ce25516d62c7/"
               "SHARE/tensorboard_log/zeroToAll")
log_dir = '/home/vvglab/tblogs' if getuser() == 'vvglab' else ozu_log_dir
settings['TENSORBOARD_LOGDIR'] = log_dir
