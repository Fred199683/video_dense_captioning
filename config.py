class Config():
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name not in self.__dict__.keys():
            self.__dict__[name] = Config()

        return self.__dict__[name]

config = Config()

config.DATASET.VOCAB_SIZE = None
config.DATASET.VOCAB_THRESHOLD = 1
config.DATASET.SEQUENCE_LENGTH = 30  # max sequence length to pad
config.DATASET.VOCAB_PATH = 'dataset/word_to_idx.json'
config.DATASET.TRAIN.IDS_PATH = 'dataset/captions/train_ids.json'
config.DATASET.TRAIN.CAPTION_PATH = 'dataset/captions/train.json'
config.DATASET.TRAIN.ENC_CAPTION_PATH = 'dataset/captions/train_encode.json'  # caption padded and encoded to indices
config.DATASET.TRAIN.FEATURE_PATH = 'dataset/features/train'
config.DATASET.VAL.IDS_PATH = 'dataset/captions/val_ids.json'
config.DATASET.VAL.CAPTION_PATH = 'dataset/captions/val_1.json'
config.DATASET.VAL.FEATURE_PATH = 'dataset/features/val'
config.DATASET.TEST.IDS_PATH = 'dataset/captions/test_ids.json'
config.DATASET.TEST.FEATURE_PATH = 'dataset/features/test'
config.DATASET.RAW_FEATURE_PATH = 'dataset/sub_activitynet_v1-3.c3d.hdf5'

config.MODEL.ERNN.ENABLE_SELECTOR = True
config.MODEL.ERNN.D_FEATURE = 500  # size of each region feature
config.MODEL.ERNN.D_HIDDEN = 1024

config.MODEL.CRNN.ENABLE_PREV2OUT = True
config.MODEL.CRNN.ENABLE_CTX2OUT = True
config.MODEL.CRNN.ENABLE_SELECTOR = True
config.MODEL.CRNN.DROPOUT = 0.5
config.MODEL.CRNN.VOCAB_LENGTH = config.DATASET.VOCAB_LENGTH
config.MODEL.CRNN.D_FEATURE = 500  # size of each region feature
config.MODEL.CRNN.D_EMBED = 300
config.MODEL.CRNN.D_HIDDEN = 1024

config.TRAIN.ENABLED = True
config.VAL.ENABLED = True
config.TEST.ENABLED = False
