from core.solver import CaptioningSolver
from core.dataset import CocoCaptionDataset
from config import config as cfg


def main():
    # load train dataset
    train_data = CocoCaptionDataset(caption_file=cfg.TRAIN.ENC_CAPTION_PATH, split='train')
    val_data = CocoCaptionDataset(caption_file=cfg.VAL.CAPTION_PATH, split='val')
    word_to_idx = train_data.get_vocab_dict()
    # load val dataset to print out scores every epoch

    solver = CaptioningSolver(word_to_idx, train_data, val_data)

    solver.train()

if __name__ == "__main__":
    main()
