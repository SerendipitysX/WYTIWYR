import argparse


parser = argparse.ArgumentParser(description='WYTIWYR')
### for classifiers
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
parser.add_argument('--epoches', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--bz', type=int, default=16, help='batchsize')
parser.add_argument('--class_num', type=int, default=18, choices=[18, 3, 2, 4])
parser.add_argument('--loss', type=str, default='Focal', choices=['CE', 'Focal'])
### for annotation and retrieval
parser.add_argument('--nu', type=int, default=1, help='the weight of explicit Intent Attributes Score')
parser.add_argument('--mu', type=int, default=5, help='the weight of implicit query(user prompt) feature')
parser.add_argument('--ip', type=str, default='10.30.11.33')
parser.add_argument('--port1', type=int, default=7779)
parser.add_argument('--port2', type=int, default=7780)
args = parser.parse_args()
