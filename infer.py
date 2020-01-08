from torch.utils.data import DataLoader
from core.solver import CaptioningSolver
from core.dataset import CocoCaptionDataset
from core.utils import load_json
from config import config as cfg
from solver import infer_collate


def main():
    cfg.TRAIN.ENABLED = False
    cfg.CHECKPOINT = 'checkpoint/model_best_meteor.pth'

    batch_size = cfg.SOLVER.INFER.BATCH_SIZE
    caption_file = cfg.DATASET.VAL.CAPTION_PATH
    is_validation = cfg.VAL.ENABLED

    # load dataset and vocab
    test_data = CocoCaptionDataset(caption_file=caption_file, config=cfg, split='val')
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, collate_fn=infer_collate)
    word_to_idx = load_json(cfg.DATASET.VOCAB_PATH)

    solver = CaptioningSolver(word_to_idx)
    solver.test(test_data, is_validation)


if __name__ == "__main__":
    main()
