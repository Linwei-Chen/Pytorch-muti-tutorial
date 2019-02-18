from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# 包含两个卷积层和两个全连接层的卷积神经网络，训练后测试的准确率在99%+
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练卷积神经网络
def train(args, model, device, train_loader, optimizer, epoch):
    # model.train()有什么用？
    # 参考 https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    """
    model.train() tells your model that you are training the model.
    So effectively layers like dropout, batchnorm etc.
    which behave different on the train and test procedures know what is going on and hence can behave accordingly.
    """
    model.train()

    # 进行训练
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    # Model使用测试模式，此时一些特殊的层比如dropout、BatchNorm等，全部按测试模式运行。
    model.eval()
    test_loss = 0
    correct = 0
    # torch.no_grad() 有什么用？
    # https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch
    # 不计算和保存梯度，节约显存
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor) 返回最大值以及其位置
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 设置命令行参数以及初始值参数
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 使生成的随机数序列一致
    # https://discuss.pytorchtutorial.com/thread-13.htm
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # pin_memory = True 可以加快CPU与GPU内存的交换
    # 但实际上使用时，我发现设置与否运算速度并没有差别
    # https://blog.csdn.net/tsq292978891/article/details/80454568
    # https://blog.csdn.net/tfcy694/article/details/83270701
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # 下面的神秘数字是啥玩意？
                           # 是train_set的均值以及标准差
                           # This is the mean and std computed on the training set.
                           # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/4
                           transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # class torchvision.transforms.Normalize(mean, std)[source]
            # Normalize a tensor image with mean and standard deviation.
            # Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
            # this transform will normalize each channel of the input torch.*Tensor
            # i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
            #
            # Parameters:
            # mean (sequence) – Sequence of means for each channel.
            # std (sequence) – Sequence of standard deviations for each channel.
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)

    # 读取已有训练好的权重
    try:
        print("loading Model...")
        model.load_state_dict(state_dict=torch.load("./mnist_cnn.pt"))
    except:
        print("loading fail")

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # 关于 amsgrad = True：
    # https://www.jiqizhixin.com/articles/2017-12-06
    # http://www.sanjivk.com/AdamConvergence_ICLR.pdf
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        # 保存模型参数
        if (args.save_model):
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
