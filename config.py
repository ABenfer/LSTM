print("ver config = 0.3.4")

# Data import
DATA_PATH = '/content/'
REAL_FEATURES_PATH = DATA_PATH + 'real_features/'
TIME_SERIES_PATH = DATA_PATH + 'time_series/'
TIME_SERIES_INTERP_PATH = DATA_PATH + 'time_series_interp/'

SCRIPT_PATH = "/content/GDrive/MyDrive/Masterarbeit/"
CHECKPOINT_PATH = SCRIPT_PATH + "Data/checkpoints/"
PLOT_PATH = SCRIPT_PATH + "Data/plots/"

N_DATA_POINTS = -1
# N_SAMPLES = 1000

# Model
LEARNING_RATE = 0.003
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 16
EPOCHS = 500
EARLY_STOPPING = 20