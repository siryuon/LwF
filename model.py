import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

import torchvision.models as models
import torchvision.transforms as transforms

def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)
    labels = torch.softmax(labels/T, dim=1)

    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    return Variable(outputs.data, requires_grad=True).cuda()

def normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
    def __init__(self, classes, classes_map, args):
        self.init_lr = args.init_lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)]
        self.lr_dec_factor = 10
        self.pretrained = False
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.epsilon = 1e-16

        super(Model, self).__init__()
        self.model = models.resnet34(pretrained=self.pretrained)
        self.model.apply(normal_init)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor)

        self.n_classes = 0
        self.n_known = 0
        self.classes_map = classes_map

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increment_classes(self, new_classes):
        n = len(new_classes)
        print('New classes: ', n)
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n
        print('New out features: ', new_out_features)

        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        self.fc = self.model.fc

        normal_init(self.fc.weight)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classifier(self, images):
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

        return preds

    def updata(self, dataset, class_map, args):
        self.compute_means = True

        prev_model = copy.deepcopy(self)
        prev_model.cuda()

        classes = list(set(dataset.train_labels))
        print('Known: ', self.n_known)

        if self.n_classes == 1 and self.n_known == 0:
            new_classes = [classes[i] for i in range(1, len(classes))]
        else:
            new_classes = [c for c in classes if class_map[c] >= self.n_known]

        if len(new_classes) > 0:
            self.increment_classes(new_classes)
            self.cuda()

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        print('Batch Size for n_classes classes: ', len(dataset))
        optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)

        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):
                for i, (indices, images, labels) in enumerate(loader):
                    seen_labels = []
                    images = Variable(torch.FloatTesnor(images)).cuda()
                    seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                    labels = Variable(seen_labels).cuda()

                    optimizer.zero_grad()
                    logits = self.forward(images)
                    cls_loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    if self.n_classes // len(new_classes) > 1:
                        dist_target = prev_model.forward(images)
                        logits_dist = logits[:, :-(self.n_classes - self.n_known)]
                        dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                        loss = dist_loss + cls_loss
                    else:
                        loss = cls_loss

                    loss.backward()
                    optimizr.step()

                    if (i+1) % 1 == 0:
                        tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data))

                    pbar.update(1)
        
