# main.py COPYRIGHT Fujitsu Limited 2021

from argparse import ArgumentParser
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from alexnet import AlexNet

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-o', '--onednn', action='store_true',
            help='use oneDNN')
    argparser.add_argument('-e', '--epochs', default=2, type=int, metavar='N',
            help='number of total epochs to run (default: 2)')
    return argparser.parse_args()


def main():
    # options
    args = get_option()
    if args.onednn:
        print('use oneDNN')

    device = "cpu"

    # Transform into Tensor, normalize RGB to mean 0.5 and 0.5 variance.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])

    # get cifar10 datasets
    dataset_path = './data'
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True,
                                     download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=dataset_path, train=False,
                                   download=True, transform=transform)

    # make DataLoader
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,)

    # load model
    model = AlexNet()
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=1e-4)

    epochs = args.epochs
    flag = args.onednn
    valid_iter = 100  # inference iterations for calculating accuracy
    with torch.backends.mkldnn.flags(enabled=flag): # True: use oneDNN
        for epoch in range(epochs):
            train(train_loader, model, device, criterion, optimizer, epoch+1, flag)
            acc = validate(val_loader, model, device, epoch+1, valid_iter)
            print('Epoch: {}/{}, Acc: {}'.format(epoch+1, epochs, acc))

    print('Acc: {}'.format(acc))


def train(train_loader, model, device, criterion, optimizer, epoch, flag):
    model.train()
    hit = 0
    total = 0

    with tqdm(train_loader, leave=False) as pbar:
        for _, (images, targets) in enumerate(pbar):
            pbar.set_description('Epoch {} Training'.format(epoch))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            outClass = outputs.cpu().detach().numpy().argmax(axis=1)
            hit += (outClass == targets.cpu().numpy()).sum()
            total += len(targets)
            pbar.set_postfix({'train Acc': hit / total * 100})


def validate(val_loader, model, device, epoch, valid_iter):
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, leave=False) as pbar:
            pbar.set_description('Epoch {} Validation'.format(epoch))
            hit = 0
            total = 0
            for i, (images, targets) in enumerate(pbar):
                outputs = model(images)
                outClass = outputs.cpu().detach().numpy().argmax(axis=1)
                hit += (outClass == targets.cpu().numpy()).sum()
                total += len(targets)
                val_acc = hit / total * 100
                pbar.set_postfix({'valid Acc': val_acc})
                if i == valid_iter:
                    break

    return val_acc


if __name__ == '__main__':
    main()
