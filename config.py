print("ver config = 0.5.9")

# Data import
DATA_PATH = '/content/'
SIMS_PATH = DATA_PATH + 'sims_interp/'

# SIMS_PATH = 'sims_interp/'

REAL_FEATURES_PATH = DATA_PATH + 'real_features/'
TIME_SERIES_PATH = DATA_PATH + 'time_series/'
TIME_SERIES_INTERP_PATH = DATA_PATH + 'time_series_interp/'

SCRIPT_PATH = "/content/GDrive/MyDrive/Masterarbeit/"
CHECKPOINT_PATH = SCRIPT_PATH + "Data/checkpoints/"
PLOT_PATH = SCRIPT_PATH + "Data/plots/"

N_DATA_POINTS = -1
# N_SAMPLES = 1000

# Model
LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = 0.995
BATCH_SIZE = 64
EPOCHS = 400
EARLY_STOPPING = 50