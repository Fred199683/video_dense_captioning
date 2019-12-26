import argparse
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.dataset import CocoCaptionDataset

parser = argparse.ArgumentParser(description='Train model.')

"""Training parameters"""
parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used for training model.')
parser.add_argument('--optimizer', type=str, default='rmsprop', help='Optimizer used to update model\'s weights.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of examples per mini-batch.')
parser.add_argument('--eval_steps', type=int, default=100, help='Evaluate and save current model every eval_steps steps.')
parser.add_argument('--metric', type=str, default='CIDEr', help='Metric being based on to choose best model, please insert on of these strings: [Bleu_i, METEOR, ROUGE_L, CIDEr] with i is 1 through 4.')
parser.add_argument('--checkpoint', type=str, help='Path to a pretrained model to initiate weights from.') 
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/', help='Path to directory where checkpoints saved every eval_steps.')
parser.add_argument('--log_path', type=str, default='log/', help='Path to directory where logs saved during the training process. You can use tensorboard to visualize logging informations and re-read IFO printed on the console in .log files.')

def main():
    args = parser.parse_args()
    # load train dataset
    train_data = CocoCaptionDataset(caption_file='./data/train/captions_train2017.json', split='train')
    val_data = CocoCaptionDataset(caption_file='./data/val/captions_val2017.json', split='val')
    word_to_idx = train_data.get_vocab_dict()
    # load val dataset to print out scores every epoch

    solver = CaptioningSolver(word_to_idx, train_data, val_data)

    solver.train(num_epochs=args.num_epochs)

if __name__ == "__main__":
    main()
