import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_ZOO_DIR = os.path.join(ROOT_DIR, 'model_zoo')
MODELS_OBJ_DETECTION_DIR = os.path.join(ROOT_DIR, 'label_maps')

IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

VIDEOS_DIR = os.path.join(ROOT_DIR, 'videos')
VIDEOS_DIR_IN = os.path.join(VIDEOS_DIR, 'inputs')
VIDEOS_DIR_OUT = os.path.join(VIDEOS_DIR, 'outputs')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_FINAL_DIR = os.path.join(DATA_DIR, 'final')

ANALYSIS_DIR = os.path.join(ROOT_DIR, 'analysis')
