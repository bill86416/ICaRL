import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

from data_loader import ifashionmnist, imnist, inotmnist
from model import iCaRLNet
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--dataset', default='mnist', help='[mnist, notmnist, fashionmnist]')
parser.add_argument('--net', default='resnet18', help='resnet18')
parser.add_argument('--trial', default='1')
#parser.add_argument('--gpu_num', type=int , default=0, help='gpu_num (default: 0)')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--debug', default=False, action='store_true')

args = parser.parse_args()
#torch.cuda.set_device(args.gpu_num)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#torch.cuda.set_device(args.gpu_num)
start_time = time.time() 

log_dir = 'log/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(log_dir + '/' + args.dataset):
    os.makedirs(log_dir + '/' + args.dataset)


test_log_file = open(log_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.net + '_cls_' + str(args.num_classes) + '_trial_' + args.trial + '.txt', "w")                
# Hyper Parameters
total_classes = 10
num_classes = args.num_classes


transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5,), (0.5,)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
])

# Initialize CNN
K = 2000 # total number of exemplars
icarl = iCaRLNet(args.net, 64, 1)
icarl = torch.nn.DataParallel(icarl, device_ids=range(torch.cuda.device_count()))
icarl.cuda()


for s in range(0, total_classes, num_classes):
    # Load Datasets
    print ("Loading training examples for classes", range(s, s+num_classes), file=test_log_file)
    print ("Loading training examples for classes", range(s, s+num_classes))
    print ("Loading testing examples for classes", range(s+num_classes), file=test_log_file)
    print ("Loading testing examples for classes", range(s+num_classes))
    
    if args.dataset == 'cifar':
        train_set = iCIFAR10(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         download=True,
                         transform=transform_test)

        test_set = iCIFAR10(root='./data',
                         train=False,
                         classes=range(num_classes),
                         download=True,
                         transform=transform_test)
    elif args.dataset == 'fashionmnist':
        train_set = ifashionmnist(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         download=True,
                         transform=transform_test)

        test_set = ifashionmnist(root='./data',
                         train=False,
                         classes=range(num_classes),
                         download=True,
                         transform=transform_test)
    elif args.dataset == 'mnist':
        train_set = imnist(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         download=True,
                         transform=transform_test)

        test_set = imnist(root='./data',
                         train=False,
                         classes=range(num_classes),
                         download=True,
                         transform=transform_test)
    elif args.dataset == 'notmnist':
        train_set = inotmnist(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         transform=transform_test)

        test_set = inotmnist(root='./data',
                         train=False,
                         classes=range(num_classes),
                         transform=transform_test)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                               shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                               shuffle=True, num_workers=4)



    # Update representation via BackProp
    #icarl.update_representation(train_set, debug=args.debug)
    icarl.module.update_representation(train_set, debug=args.debug)
    m = int(K / icarl.module.n_classes)
    print(m)
    print(icarl.module.n_classes)
    # Reduce exemplar sets for known classes
    icarl.module.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in range(icarl.module.n_known, icarl.module.n_classes):
        print ("Constructing exemplar set for class-%d..." %(y), file=test_log_file)
        images = train_set.get_image_class(y)
        icarl.module.construct_exemplar_set(images, m, transform_test)
        print ("Done")


    for y, P_y in enumerate(icarl.module.exemplar_sets):
        print ("Exemplar set for class-%d:" % (y), P_y.shape, file=test_log_file)
        if args.debug:
            break        

       #show_images(P_y[:10])

    icarl.module.n_known = icarl.module.n_classes
    print ("iCaRL classes: %d" % icarl.module.n_known, file=test_log_file)

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.module.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()
        if args.debug:
            break        


    print('Train Accuracy: %.3f %%' % (100 * correct / total), file=test_log_file)

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.module.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()
        if args.debug:
            break        

    print('Test Accuracy: %.3f %%' % (100 * correct / total), file=test_log_file)

    #if args.debug:
    #    break        

elapsed_time = time.time() - start_time
print('Time %.3f' % (elapsed_time), file=test_log_file)
