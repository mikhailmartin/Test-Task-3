import os


RANDOM_STATE = 42

TRAIN_DATA_PATH = os.path.join(os.getcwd(), '..', 'data', 'raw', 'train.pkl')
TEST_DATA_PATH = os.path.join(os.getcwd(), '..', 'data', 'raw', 'test.pkl')

LGBM_MODEL_PATH = os.path.join(os.getcwd(), '..', 'model', 'lgbm_model.pkl')
