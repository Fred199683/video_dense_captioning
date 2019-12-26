from core.solver import CaptioningSolver
from core.dataset import CocoCaptionDataset
from config import config as cfg
from core.utils import load_json


def main():
    # load train dataset
    train_data = CocoCaptionDataset(caption_file=cfg.DATASET.TRAIN.ENC_CAPTION_PATH, split='train')
    val_data = CocoCaptionDataset(caption_file=cfg.DATASET.VAL.CAPTION_PATH, split='val')
    word_to_idx = load_json(cfg.DATASET.VOCAB_PATH)
    # load val dataset to print out scores every epoch

    solver = CaptioningSolver(word_to_idx, train_data, val_data)

    solver.train()

if __name__ == "__main__":
    main()
