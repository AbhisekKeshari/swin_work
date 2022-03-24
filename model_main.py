import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import json

""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


# from model.model_main import IQARegression
# from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
from model.swin_class import SaveOutput, swin_transformer
from model.trainer import train_epoch, eval_epoch
# from trainer import train_epoch, eval_epoch
from util import RandCrop, RandHorizontalFlip, RandRotation, Normalize, ToTensor, RandShuffle

# config file
config = Config({
    # device
    "GPU_ID": "0",
    "num_workers": 8,

    # model for PIPAL (NTIRE2022 Challenge)
    "n_enc_seq": 21 * 21,  # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 21 * 21,  # feature map dimension (H x W) from backbone
    "n_layer": 1,  # number of encoder/decoder layers
    "d_hidn": 128,  # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,  # feed forward hidden layer dimension
    "d_MLP_head": 128,  # hidden layer of final MLP
    "n_head": 4,  # number of head (in multi-head attention)
    "d_head": 128,  # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,  # dropout ratio of transformer
    "emb_dropout": 0.1,  # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,  # dimension of final prediction
    "crop_size": 224,  # input image crop size

    "model_name": "swin_large_patch4_window7_224_in22k",

    # data
    "db_name": "PIPAL",  # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
    "db_path": "/home/ubuntu/Documents/Dataset/Train_Images",   # root of dataset
    "snap_path": "./weights/swin_train",  # path for saving weights
    "snap_path1": "./weights/swin_eval",
    "txt_file_name": "/home/ubuntu/Desktop/pythonProject/dataset/dataset_txt_files/PIPAL.txt",  # image list file (.txt)
    "train_size": 0.9,
    "scenes": "all",

    # ensemble in validation phase
    "test_ensemble": True,
    "n_ensemble": 5,

    # optimization
    "batch_size": 16,
    "learning_rate": 1e-6,
    "weight_decay": 1e-5,
    "n_epoch": 300,
    "val_freq": 1,
    "save_freq": 5,
    "checkpoint": None,  # load pretrained weights
    "T_max": 50,  # cosine learning rate period (iteration)
    "eta_min": 0  # mininum learning rate
})

# device setting
config.device = torch.device("cuda:%s" % config.GPU_ID if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % config.GPU_ID)
else:
    print('Using CPU')

# data selection
from dataset.data_PIPAL import IQADataset

# data separation (8:2)
train_scene_list, test_scene_list = RandShuffle(config.scenes, config.train_size)

print('number of train scenes: %d' % len(train_scene_list))
print('number of test scenes: %d' % len(test_scene_list))

# data load
train_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform=transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), RandHorizontalFlip(), RandRotation(), ToTensor()]),
    train_mode=True,
    scene_list=train_scene_list,
    train_size=config.train_size
)
test_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform= transforms.Compose([Normalize(0.5, 0.5), ToTensor()]) if config.test_ensemble else transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), ToTensor()]),
    train_mode=False,
    scene_list=test_scene_list,
    train_size=config.train_size
)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                          drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,drop_last=True, shuffle=True)

# create model
model = swin_transformer(config.model_name,True)
# model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background').to(config.device)

# save intermediate layers
save_output = SaveOutput()
hook_handles = []




for layer in model.modules():
    if(layer ==model.model.layers[3]):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
        print(layer)
        break

# loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

def saveModel(epoch, loss,  xt, xo, xs):
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path1, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': xt,
            'optimizer_state_dict': xo,
            'scheduler_state_dict': xs,
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))
# train & validation
losses, scores = [], []
i=1
slcc=0
plcc=0
# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
losses, scores = [], []
for epoch in range(start_epoch, config.n_epoch):
    loss, rho_s, rho_p, xt, xo, xs = train_epoch(config, epoch, model, save_output, criterion,
                                     optimizer, scheduler, train_loader)
    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

        # print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))
    if (epoch+1) % config.val_freq == 0:
        loss_v, rho_s, rho_p = eval_epoch(config, epoch, model, save_output, criterion,
                                     optimizer, scheduler, train_loader)
        print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))
        if (i == 1):
            saveModel(epoch, loss, xt, xo, xs)
            score = rho_s + rho_p
            i = i + 1
        elif (score < (rho_s + rho_p)):
            saveModel(epoch, loss, xt, xo, xs)
            score = rho_s + rho_p
        else:
            continue
