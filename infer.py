from core.solver import CaptioningSolver
from core.dataset import CocoCaptionDataset
from core.utils import load_json
from config import config as cfg


def main():
    cfg.TRAIN.ENABLED = False
    cfg.CHECKPOINT = 'checkpoint/model_best_meteor.pth'

    caption_file = cfg.DATASET.VAL.CAPTION_PATH
    is_validation = cfg.VAL.ENABLED

    # load dataset and vocab
    test_data = CocoCaptionDataset(caption_file=caption_file, config=cfg, split='val')
    word_to_idx = load_json(cfg.DATASET.VOCAB_PATH)

    solver = CaptioningSolver(word_to_idx)
    solver.test(test_data, is_validation)


if __name__ == "__main__":
    main()
