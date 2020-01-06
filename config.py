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
config.DATASET.SEQUENCE_LENGTH = 51                                             # max sequence length to pad and eliminate those sentences exceed it
config.DATASET.VOCAB_PATH = 'dataset/word_to_idx.json'                          # vocabulary to be saved by prepro.py and used by train.py and infer.py
config.DATASET.TRAIN.IDS_PATH = 'dataset/captions/train_ids.json'               # training ids json
config.DATASET.TRAIN.CAPTION_PATH = 'dataset/captions/train.json'               # training caption json
config.DATASET.TRAIN.ENC_CAPTION_PATH = 'dataset/captions/train_encode.json'    # preprocessed caption file to be saved by prepro.py and used by train.py and infer.py
config.DATASET.TRAIN.FEATURE_PATH = 'dataset/features/train'                    # prepocessed feature file to be saved by prepro.py and used by train.py and infer.py
config.DATASET.VAL.IDS_PATH = 'dataset/captions/val_ids.json'                   # training ids json
config.DATASET.VAL.CAPTION_PATH = 'dataset/captions/val_1.json'                 # training caption json
config.DATASET.VAL.FEATURE_PATH = 'dataset/features/val'                        # prepocessed feature file to be saved by prepro.py and used by train.py and infer.py
config.DATASET.TEST.IDS_PATH = 'dataset/captions/test_ids.json'                 # training ids json
config.DATASET.TEST.FEATURE_PATH = 'dataset/features/test'                      # training caption json
config.DATASET.RAW_FEATURE_PATH = 'dataset/sub_activitynet_v1-3.c3d.hdf5'       # raw video feature data

config.MODEL.ERNN.ENABLE_SELECTOR = True  # set to True to enable a selecting gate of context before being fed to RNN
config.MODEL.ERNN.D_FEATURE = 500         # feature size
config.MODEL.ERNN.D_HIDDEN = 1024         # RNN hidden size

config.MODEL.CRNN.ENABLE_PREV2OUT = True  # set to True to directly link previous hidden state to current output
config.MODEL.CRNN.ENABLE_CTX2OUT = True   # set to True to directly link context to current output
config.MODEL.CRNN.ENABLE_SELECTOR = True  # set to True to enable a selecting gate of context before being fed to RNN
config.MODEL.CRNN.DROPOUT = 0.5           # dropout proportion of RNN
config.MODEL.CRNN.D_FEATURE = 500         # feature size
config.MODEL.CRNN.D_EMBED = 300           # embedding size
config.MODEL.CRNN.D_HIDDEN = 1024         # RNN hidden size

config.SOLVER.TRAIN.OPTIM = 'adam'              # optimizer
config.SOLVER.TRAIN.LR = 0.001                  # learning rate
config.SOLVER.TRAIN.N_EPOCHS = 50               # number of epochs
config.SOLVER.TRAIN.BATCH_SIZE = 16             # batch size being used while training
config.SOLVER.TRAIN.EVAL_STEPS = 500            # steps taken before evaluation during training
config.SOLVER.TRAIN.CKPT = None                 # input path to a checkpoint if you want to resume training from it
config.SOLVER.TRAIN.CKPT_DIR = 'checkpoint/'    # path to dir saving checkpoints during training
config.SOLVER.TRAIN.LOG_DIR = 'log/'            # path to logging dir
config.SOLVER.TRAIN.ALPHA_C = 1.0               # param for training a constrain to alphas, make the attention spread over entire sequence (caption).

config.SOLVER.INFER.N_TIME_STEPS = 30           # number of time steps in inference
config.SOLVER.INFER.BEAM_SIZE = 3               # beam size being used while inference
config.SOLVER.INFER.LEN_NORM = 0.4              # length normalization being used while inference
config.SOLVER.INFER.RESULT_PATH = 'results/results.json'  # path to save results after inference
config.SOLVER.INFER.EVAL_PATH = 'results/eval.json'  # path to save eval results after evaluation

config.SOLVER.CAPTURED_METRICS = ['meteor', 'cider']

config.DEVICE = 'cuda:0'  # decide which gpu being used

config.TRAIN.ENABLED = True  # enable training phrase to be processed in prepro
config.VAL.ENABLED = True    # enable validating phrase to be processed in prepro or train
config.TEST.ENABLED = False  # enable testing phrase to be processed in prepro or train
