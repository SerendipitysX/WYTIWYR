import torch.utils as utils
import torch.nn as nn
import random
import tqdm
import torch.optim as optim
import sys
import os
current_path = os.path.dirname(os.getcwd())
print(current_path)
sys.path.append(current_path + '/utils')
sys.path.append(current_path + '/classifier')
from utils.dataloader import *
from utils.loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from backbone import get_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

layout = {
    'PLOT': {
        "loss": ["Multiline", ['Loss/train_self', 'Loss/train_clip', 'Loss/val']],
        "Metrics": ["Multiline", ['Metrics/acc', 'Metrics/lr']],
    },
}
writer = SummaryWriter()
writer.add_custom_scalars(layout)


# =============================== environment ================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False ## find fittest convolution
    torch.backends.cudnn.deterministic = True ## keep experiment result stable
set_env(seed=args.seed)

# =============================== dataset ================================
# all data
annotation = pd.read_csv('./data/layout_annotation.csv')
all_data = all_Dataset(annotations_file=annotation, img_dir='./data/all/', transform=transform, all=True)
print(len(all_data))
train_idx, valid_idx, test_idx = split_dataset(all_data)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_dataloader = torch.utils.data.DataLoader(all_data, batch_size=args.bz, shuffle=False, sampler=train_sampler, num_workers=8, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(all_data, batch_size=args.bz, shuffle=False, sampler=valid_sampler, num_workers=8, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(all_data, batch_size=args.bz, shuffle=False, sampler=test_sampler, num_workers=8, drop_last=True)


# =============================== model ==============================
model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
model_name = model_names[0]
model = get_model(model_name, class_num=3).to(device)


# ========================== optimizer & loss ========================
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

if args.loss == 'CE':
    print(args.loss)
    loss = nn.CrossEntropyLoss()
if args.loss == 'Focal':
    print(args.loss)
    loss = FocalLoss()

# =============================== train ================================
def train(model, train_iter, val_iter, epochs, optimizer, scheduler, loss):
    min_val_loss = np.inf
    min_acc = 0
    for epoch in range(epochs):
        model.train()
        l_sum = []
        for (images, img_paths, labels) in tqdm(train_iter):
            images, labels = images.to(device).to(torch.float32), labels.to(device).to(torch.int64)
            #  ## get y_pred and some other probs
            y_pred = model(images)  # (bz,17)?
            #  ## loss
            l_self = loss(y_pred, labels)  # (bz, 1)
            loss_total = l_self
            writer.add_scalar("Loss/train_self", np.mean(l_self.item()), epoch)
            # ## optim
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            # ## log
            l_sum.append(np.mean(loss_total.item()))
        scheduler.step()
        l_epoch = np.mean(l_sum)
        val_loss, acc = val(val_model=model, val_iter=val_iter, epoch=epoch)
        writer.add_scalar("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Metrics/acc", acc, epoch)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.6f} |Train loss: {:.6f} | Val loss: {:.6f} | Acc: {:.4f} |GPU occupy: {:.6f} MiB'.\
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_epoch, val_loss, acc, gpu_mem_alloc))
        if acc > min_acc:
            min_acc = acc
            print("model saved!")
            torch.save(model.state_dict(), './models/layout/'+model_name+'_model_only_CE.pth')

    print('\nTraining finished.\n')


def val(val_model, val_iter, epoch):
    model.eval()
    l_epoch, acc_epoch = [], []
    acc = 0
    total = 0
    with torch.no_grad():
        for (images, img_paths, labels) in tqdm(val_iter):
            images, labels = images.to(device), labels.to(device).to(torch.int64)
            y_pred = val_model(images)  # (bz,17)?
            y_pred_softmax = torch.nn.functional.softmax(y_pred, dim=1)  # (bz,17)
            y_pred_vector = y_pred_softmax.gather(1, labels.view(-1, 1))  # (bz, 1)

            #  ## loss
            l_self_v = loss(y_pred, labels)  # (bz, 1)
            loss_total_v = l_self_v

            #  ## acc
            _, max_idx = torch.max(y_pred, 1)
            tmp = max_idx - labels

            # print(torch.count_nonzero(tmp))
            correct_num = len(labels) - torch.count_nonzero(tmp).cpu()
            #  ## log
            l_epoch.append(np.mean(loss_total_v.item()))
            acc += correct_num
            total += len(max_idx)

        l_epoch = round(np.average(l_epoch), 2)
        acc_rate = acc/total
        return l_epoch, acc_rate


if __name__ == '__main__':
    set_env(args.seed)
    train(model, train_dataloader, valid_dataloader, args.epoches, optimizer, scheduler, loss)
    writer.flush()
