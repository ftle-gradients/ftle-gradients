import os

import matplotlib.pyplot as plt
import torch
from config import *
import numpy as np
from models import *
from tqdm import tqdm
from training import *
import time
import imageio
import torchvision as T
import argparse
# from sMNIST import Model#, test_model
from utils import select_network, select_optimizer
from SMNIST_FTLE import FTLE
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

## Args
parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN',
                    choices=['RNN', 'nnRNN', 'LSTM', 'expRNN'],
                    help='options: RNN, nnRNN, expRNN, LSTM')
parser.add_argument('--nhid', type=int,
                    default=512,
                    help='hidden size of recurrent net')
parser.add_argument('--cuda', action='store_true',
                    default=False, help='use cuda')
parser.add_argument('--random-seed', type=int,
                    default=400, help='random seed')
parser.add_argument('--permute', action='store_true',
                    default=True, help='permute the order of sMNIST')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--save-freq', type=int,
                    default=50, help='frequency to save data')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--optimizer', type=str, default='RMSprop',
                    choices=['Adam', 'RMSprop'],
                    help='optimizer: choices Adam and RMSprop')
parser.add_argument('--alpha', type=float,
                    default=0.99, help='alpha value for RMSprop')
parser.add_argument('--betas', type=tuple,
                    default=(0.9, 0.999), help='beta values for Adam')

parser.add_argument('--rinit', type=str, default="xavier",
                    choices=['random', 'cayley', 'henaff', 'xavier'],
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier",
                    choices=['xavier', 'kaiming'],
                    help='input weight matrix initialization')
parser.add_argument('--nonlin', type=str, default='tanh',
                    choices=['none', 'modrelu', 'tanh', 'relu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--alam', type=float, default=0.0001,
                    help='decay for gamma values nnRNN')
parser.add_argument('--Tdecay', type=float,
                    default=0, help='weight decay on upper T')

args = parser.parse_args()


def inverse_permutation(perm):
    perm_tensor = torch.LongTensor(perm)
    inv = torch.empty_like(perm_tensor)
    inv[perm_tensor] = torch.arange(perm_tensor.size(0), device=perm_tensor.device)
    return inv


lr = args.lr
lr_orth = args.lr_orth
random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
SAVEFREQ = args.save_freq
inp_size = 1
alam = args.alam
Tdecay = args.Tdecay
permute = True

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
# device = torch.device('cpu')
le_batch_size = 15
hidden_size = args.nhid
output_size = 10

torch.cuda.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if permute:
    rng = np.random.RandomState(1234)
    order = rng.permutation(784)
else:
    order = np.arange(784)

inverse_order = inverse_permutation(order)


class Model(nn.Module):
    def __init__(self, hidden_size, rnn, eval=True):
        super(Model, self).__init__()
        self.rnn_layer = rnn
        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size, 10)
        self.loss_func = nn.CrossEntropyLoss()
        self.evaluate = eval

    def forward(self, inputs, y=None, order=None, h=None):
        inputs = inputs[:, order]
        inputs = inputs.squeeze(dim=1)
        for input in torch.unbind(inputs, dim=1):
            h = self.rnn_layer(input.view(-1, inp_size), h)
        out = self.lin(h)

        if self.evaluate:
            loss = self.loss_func(out, y)
            preds = torch.argmax(out, dim=1)
            correct = torch.eq(preds, y).sum().item()
            return loss, correct
        else:
            return out, h


def test_model(net, dataloader, order):
    accuracy = 0
    loss = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data
            x = x.view(-1, 784)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            if NET_TYPE == 'LSTM':
                net.rnn_layer.init_states(x.shape[0])
            loss, c = net.forward(x, y, order)
            accuracy += c
    accuracy /= len(testset)
    return loss, accuracy


# h_0 = torch.randn(1, le_batch_size, args.nhid).to(device)
h_0 = torch.zeros(1, le_batch_size, args.nhid).to(device)
# targets = testset.targets[:trunc].view(le_batch_size, -1, 1)

le_loader_fname = f'SMNIST/le_loader_b{le_batch_size}.p'
if os.path.exists(le_loader_fname):
    print('Loading Dataloader')
    le_loader, le_input, le_target = torch.load(le_loader_fname)
else:
    print('Saving Dataloader')
    testset = T.datasets.MNIST(root='./MNIST',
                               train=False,
                               download=True,
                               transform=T.transforms.ToTensor())
    le_input = testset.data[:le_batch_size].view(-1, 784, inp_size) / 255
    le_target = testset.targets[:le_batch_size].view(-1)
    le_dataset = torch.utils.data.TensorDataset(le_input, le_target)

    le_loader = torch.utils.data.DataLoader(le_dataset,
                                            batch_size=le_batch_size)
    torch.save((le_loader, le_input, le_target), le_loader_fname)

rnn = select_network(args, inp_size)
rnn.batch_first = True
model = Model(hidden_size, rnn).to(device)
# model.rnn_layer.batch_first = False
epoch = 50
model_name = f'RNN_{epoch}.pth.tar'
best_state = torch.load(f'SMNIST/Models/{model_name}')
model.load_state_dict(best_state['state_dict'])

model.evaluate = False

hidden_size = 512
input_seq_length = 784
grads = True
r_plot = False
r_thresh = True
calc_grads = False
load_grads = True
le_load = True
plot_grads = False

if permute:
    perm_suffix = '_permuted'
    perm_plot = ', Permuted'
else:
    perm_suffix = ''
    perm_plot = ''
fname = f'SMNIST/ftle_dict_{args.nonlin}{perm_suffix}_b{le_batch_size}_e{epoch}.p'

model.evaluate = False

if os.path.exists(fname) and le_load:
    print('Loading FTLEs')
    ftle_dict = torch.load(fname, map_location=device)
    print('FTLES Loaded')
else:
    print('Calculating FTLEs')
    ftle_dict = FTLE(le_input[:, order], h_0, model=model, rec_layer='rnn')
    torch.save(ftle_dict, fname)

if r_thresh:
    torch.manual_seed(31)
    plot_dir = f'SMNIST/Plots/rthresh/Epoch{epoch}'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    suffix = ''

    grads_fname = f'SMNIST/Grads/trainedRNN_grads{perm_suffix}_e{epoch}_b{le_batch_size}.p'
    logits_fname = f'SMNIST/Grads/trainedRNN_logits{perm_suffix}_e{epoch}_b{le_batch_size}.p'

    pred_list = torch.load(logits_fname)
    pred_logits = pred_list.softmax(dim=-1)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    base_loss = criterion(pred_list[-1], le_target)
    # best_idx = torch.argmin(criterion(pred_list[-1], le_target))

    r_vals = ftle_dict['rvals']
    x = torch.arange(r_vals.shape[1])
    plot_evolution = False
    # n_ev_plots = 5
    plot_flips = False
    plot_input = False

    # Flip pixels based on mean value of first n FTLEs
    n_list = [1, 5, 10, 50, 100, 250, 512]
    k_list = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 200, 300]

    # flipped_losses = torch.zeros(len(k_list), len(n_list), le_batch_size)
    # flip_counts = torch.zeros(len(k_list), len(n_list), 2, le_batch_size)
    # flip_le_inputs = torch.zeros(len(k_list), len(n_list), le_batch_size, input_seq_length, inp_size)

    # Store Random Flips
    no_randoms = 50
    random_losses = torch.empty(len(k_list), len(n_list), no_randoms, le_batch_size)
    random_inputs = torch.empty(len(k_list), len(n_list), no_randoms, le_batch_size, input_seq_length, inp_size)

    sorted_rvals_list = torch.empty(len(n_list), le_batch_size, input_seq_length)
    sorted_orders_list = torch.empty(len(n_list), le_batch_size, input_seq_length)

    n_select = 0
    k_select = 9
    idcs_select = [0, 11, 12, 13, 14]
    n = 1
    n_idx = 0
    k = 100
    k_idx = 9

    k_label = 'first'
    thresh_dir = f'{plot_dir}/Rvals{n}'
    rvals_fname = f'{thresh_dir}/rvals_sorted_b{le_batch_size}.p'
    thresh_suffix = f'_{k_label}'
    flipped_fname = f'{thresh_dir}/flipped_losses_k{k}{thresh_suffix}.p'
    flipped_loss, flipped_le_input, flip_count = torch.load(flipped_fname)

    rand_flip_fname = f'{thresh_dir}/rand_n{no_randoms}_k{k}{thresh_suffix}.p'
    rand_losses, rand_flipped_inputs = torch.load(rand_flip_fname)

    plot_input = False
    if plot_input:
        in_plot_dir = f'SMNIST/Plots/rthresh/Epoch{epoch}/Inputs'
        if not os.path.exists(in_plot_dir):
            os.mkdir(in_plot_dir)
        for idx_select in idcs_select:
            original_input = le_input[idx_select]
            plt.figure(figsize=(3, 3))
            plt.imshow(original_input.reshape(28, 28))
            plt.savefig(f'{in_plot_dir}/original_input_{idx_select}_k{k_select}.png', dpi=400, bbox_inches='tight')

            # for k_label in k_labels:
            plt.figure(figsize=(3, 3))
            flipped_input = flip_le_inputs[k_select, n_select, idx_select]
            plt.imshow(flipped_input.reshape(28, 28))
            plt.savefig(f'{in_plot_dir}/flipped_input_{idx_select}_k{k_select}.png', dpi=400, bbox_inches='tight')

    plot_logits = True
    n_select = [1]
    k_select = [100]
    rand_idcs = [1, 3, 5, 7, 9]
    versions = 5
    num_plotted = 5
    styles = [(0, (1, 1)), (0, (8, 5)), (0, (3, 5, 3, 5)), (0, (5, 1)), (0, (.5, .5))]
    input_idcs = [4, 12]

    k_dir = f'{plot_dir}/Rvals{n}/K{k}'
    if not os.path.exists(k_dir):
        os.mkdir(k_dir)

    for input_idx in tqdm(input_idcs):
        df_fname = f'SMNIST/Plots/rthresh/Epoch{epoch}/Rvals{n}/all_logits_idx{input_idx}.p'
        if os.path.exists(df_fname):
            data = torch.load(df_fname)
        else:
            data = torch.empty(10, no_randoms + 2)
            data[:, 0] = pred_logits[-1, input_idx].T
            f_logits = model(flipped_le_input[input_idx, order].unsqueeze(0), h_0)[0].softmax(dim=-1).detach().t()
            rand_logits = model(rand_flipped_inputs[:, input_idx, order], h_0)[0].softmax(dim=-1).detach().t()
            data[:, 1] = f_logits.squeeze()
            data[:, 2:] = rand_logits
            torch.save(data, df_fname)
        base_names = ['Original', 'Rvals']
        rand_names = [f'Random{r}' for r in (range(1, no_randoms + 1))]
        col_names = base_names + rand_names
        df = pd.DataFrame(data=data, columns=col_names)
        rand_samples = [1, 2, 3]
        rand_col_names = [f'Random{r}' for r in rand_samples]

        no_randoms_plot = 5
        # oranges = [(255, 100, 0), (255, 175, 0)]
        o_value = np.linspace(80, 250, num=no_randoms_plot)/255.
        # cm = LinearSegmentedColormap.from_list(
        #     "Custom", oranges, N=no_randoms)
        plt.figure(figsize=(3,3))
        X = torch.arange(10)
        bar_width = 0.2
        num_bars = 4
        label_width = 1
        if input_idx == 4:
            rands = [6,1,2, 7,5]
        elif input_idx == 12:
            rands = [0,1,5, 11, 10]
        # offsets = [-bar_width/2 + (-num_bars/2 + i)*bar_width for i in range(num_bars)]
        plt.bar(X, df['Original'], 0.8, label='Original', color='k', alpha=0.8)
        plt.bar(X, df['Rvals'], 0.8, label='Rvals', color='b', alpha=0.8)
        # plt.bar(X - 0.4, df[f'Random{rands[0]+1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[0], 0), alpha = 0.9)
        # plt.bar(X - 0.2, df[f'Random{rands[1]+1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[1], 0), alpha = 0.9)
        # plt.bar(X + 0, df[f'Random{rands[2]+1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[2], 0), alpha = 0.9)
        # plt.bar(X + 0.2, df[f'Random{rands[3] + 1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[3], 0), alpha = 0.9)
        # plt.bar(X + 0.4, df[f'Random{rands[4] + 1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[4], 0), alpha = 0.9)
        plt.bar(X - 0.3, df[f'Random{rands[0] + 1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[0], 0),
                alpha=0.9)
        plt.bar(X - 0.1, df[f'Random{rands[1]+1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[1], 0), alpha = 0.9)
        plt.bar(X + 0.1, df[f'Random{rands[2]+1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[2], 0), alpha = 0.9)
        plt.bar(X + 0.3, df[f'Random{rands[3] + 1}'], bar_width, label=f'Random{rands[0]}', color=(1, o_value[3], 0), alpha = 0.9)
        plt.xlim([-0.5, 9.5])
        plt.xticks(X)
        # plt.show()
        # plt.xlabel('Label')
        # plt.ylabel('Logit Value')
        plt.savefig(f'{k_dir}/logits_bar_idx{input_idx}_thick.png', dpi=400, bbox_inches='tight')
