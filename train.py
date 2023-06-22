import os
import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_data, get_model, train, evaluate, set_seed



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', help='folder path of training data')
    parser.add_argument('-test', help='folder path of test data')
    parser.add_argument('-tr_txt', help='txt file of training data')
    parser.add_argument('-test_txt', help='txt file of test data')
    parser.add_argument('-epoch', type=int, help='number of training epoch')
    parser.add_argument('-bs', type=int, help='batch size')
    parser.add_argument('-size', type=int, help='image size (an int)')
    parser.add_argument('-lr', type=float, help='learn rate')
    parser.add_argument('-wd', type=float, default=0, help='weight decay factor')
    parser.add_argument('-save_dir', default='weight', help='the path where model will be saved (default: weight)')
    parser.add_argument('-seed', type=int, help='random seed')
    return parser.parse_args()


def main():
    # arguments
    args = get_args()

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    # create folder for model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # data
    tr_ds, val_ds, test_ds = load_data(args.tr,
                                       args.tr_txt, 
                                       args.test,
                                       args.test_txt,
                                       args.size,
                                       args.bs, 
                                       seed=args.seed)
    
    # model ...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device)
    opt = Adam(model.parameters(), args.lr, weight_decay=args.wd)
    schr = ReduceLROnPlateau(opt, factor=0.1, patience=2, verbose=True)
    loss_fn = nn.MSELoss().to(device)

    # print
    print('=' * 25)
    print('device:', device)
    print('train data:', len(tr_ds.dataset))
    print('val data:', len(val_ds.dataset))
    print('test data:', len(test_ds.dataset))
    print('=' * 25)

    # train
    best_model = train(model, tr_ds, val_ds, 
                       device, opt, schr, loss_fn, 
                       args.epoch, args.save_dir)
    
    # test
    test_loss = evaluate(best_model, test_ds, device, loss_fn)
    print(f'test loss: {test_loss:.4f},  rmse: {test_loss.pow(0.5):.4f}')



if __name__ == '__main__':
    main()