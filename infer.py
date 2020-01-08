from torch.utils.data import DataLoader
from core.solver import CaptioningSolver, infer_collate
from core.dataset import CocoCaptionDataset
from core.utils import load_json, evaluate
from config import config as cfg


def main():
    cfg.TRAIN.ENABLED = False
    cfg.VAL.ENABLED = False
    cfg.SOLVER.CHECKPOINT = 'checkpoint/model_best_meteor.pth'

    batch_size = cfg.SOLVER.INFER.BATCH_SIZE
    caption_file = cfg.DATASET.VAL.CAPTION_PATH
    result_file = cfg.SOLVER.INFER.RESULT_PATH
    is_validation = cfg.VAL.ENABLED

    # load dataset and vocab
    test_data = CocoCaptionDataset(caption_file=caption_file, config=cfg, split='val')
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, collate_fn=infer_collate)
    word_to_idx = load_json(cfg.DATASET.VOCAB_PATH)

    solver = CaptioningSolver(word_to_idx)
    solver.test(test_loader, is_validation)

    evaluate(result_file, verbose=True)


if __name__ == "__main__":
    main()
