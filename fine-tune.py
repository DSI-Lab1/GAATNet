from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score

from model import Model
from utils import load_data, make_one_hot, compute_contrastive_loss

# fine-tuning settings
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_dataset', type=str, default='Computers')
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--mode', type=str, default='finetune', help='use adapter or not')
parser.add_argument('--adapter_size', type=int, default=8, help='adapter size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=201, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--gamma_dimension', type=int, default=8, help='Dimension of learnable weight matrices gamma.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--Beta', type=float, default=0.3, help='Beta')
parser.add_argument('--alpha2', type=float, default=0.4, help='Restart probability for controlling diffusion intensity')
parser.add_argument('--embedding_dim', type=int, default=256, help='Node feature dimensions')
parser.add_argument('--lambda0', type=float, default=0.4, help='Controlling the weight of Contrastive Loss')
parser.add_argument('--temperature', type=float, default=0.5, help='Temperature parameters for Contrastive Loss')
args = parser.parse_args()
dataset = args.dataset
args.cuda = not args.no_cuda and torch.cuda.is_available()
beta = args.Beta
EMBEDDING_DIM = args.embedding_dim

# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
t_total = time.time()
adj, features, labels_train, labels_test, idx_train, idx_test = load_data(dataset, EMBEDDING_DIM)

# load pretrained model, Model and optimizer
model = Model(mode=args.mode, nfeat=features.shape[1], nhid=args.hidden, ndimension=args.gamma_dimension,
                dropout=args.dropout, nheads=args.nb_heads, alpha=args.alpha)
model.train_adapter()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
dataset_folder = f"pretrained_{args.pretrained_dataset}"
save_path = os.path.join(dataset_folder, f"best_model_and_optimizer_{args.pretrained_dataset}_AUC0.92_F-score0.85.pt")
if os.path.exists(save_path):
    print(f"Loading model from: {save_path}")
    checkpoint = torch.load(save_path)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

# Initialization
pre_output = torch.zeros((labels_train.shape[0]))
labels_one_hot = make_one_hot(labels_train)

# Cuda
if args.cuda:
    model.cuda()
    adj = adj.cuda()
    features = features.cuda()
    pre_output = pre_output.cuda()
    labels_one_hot = labels_one_hot.cuda()
    idx_train = idx_train.cuda()
    labels_train = labels_train.cuda()
    idx_test = idx_test.cuda()
    labels_test = labels_test.cuda()


# Train model
loss_train_values = []
f1_train_values = []
auc_values = []
f1_values = []
best_pred_train_f1_score = 0
bad_counter = 0
mini_loss = float('inf')
train_times = []
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_train = model(features, adj, idx_train[:,0], idx_train[:,1])
    link_pred_loss = torch.mean(F.binary_cross_entropy(output_train, labels_train), axis=0)
    contrastive_loss = compute_contrastive_loss(features, idx_train, labels_train, args.temperature)
    loss = link_pred_loss + args.lambda0 * contrastive_loss

    loss.backward()
    optimizer.step()
    loss_train_values.append(loss.data.item())
    pred_train_f1_score = f1_score(labels_train.detach().numpy(), (output_train.detach().numpy() > 0.5).astype(int))
    f1_train_values.append(pred_train_f1_score)

    # test
    model.eval()
    with torch.no_grad():
        output_test = model(features, adj, idx_test[:,0], idx_test[:,1])
        auc = roc_auc_score(labels_test.detach().numpy(), output_test.detach().numpy())
        predictions = (output_test.detach().numpy() > 0.5).astype(int)
        f1 = f1_score(labels_test.detach().numpy(), predictions)

        auc_values.append(auc)
        f1_values.append(f1)

    train_times.append(time.time() - t)


    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss.data.item()),
          'test AUC {:.4f}'.format(auc),
          'test F1 score {:.4f}'.format(f1),
          'time: {:.4f}s'.format(time.time() - t))

    # early stop
    if loss < mini_loss:
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Max AUC: {:.4f}".format(max(auc_values)),
      "Max F1 score: {:.4f}".format(max(f1_values)))
print("Avg Times： {:.4f}s".format(sum(train_times) / len(train_times)))



