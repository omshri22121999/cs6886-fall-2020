import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import wandb

run = int(open("runs.txt", "r").read())

# Hsigmoid Implementation
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


# SEModule Implementation
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# H-Swish Implementation
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


# Fuse Layer
class Fuse(nn.Module):
    def __init__(
        self,
        in_channel,
        exp_channel,
        out_channel,
        non_linearity,
        kernel_size,
        stride,
        is_se=True,
        apply_bn=True,
    ):
        super(Fuse, self).__init__()
        if non_linearity == "relu":
            self.nl = nn.ReLU(inplace=True)
        elif non_linearity == "hswish":
            self.nl = Hswish(inplace=True)
        else:
            raise Exception("Please use proper non-linearity")
        self.is_se = is_se
        self.apply_bn = apply_bn

        # Defining trainable parameters

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=exp_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.batchn1 = nn.BatchNorm2d(num_features=exp_channel)

        self.conv2_a = nn.Conv2d(
            in_channels=exp_channel,
            out_channels=exp_channel,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=((kernel_size - 1) // 2, 0),
            groups=exp_channel,
            bias=False,
        )
        self.conv2_b = nn.Conv2d(
            in_channels=exp_channel,
            out_channels=exp_channel,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, (kernel_size - 1) // 2),
            groups=exp_channel,
            bias=False,
        )

        self.batchn2_a = nn.BatchNorm2d(num_features=exp_channel)
        self.batchn2_b = nn.BatchNorm2d(num_features=exp_channel)

        if self.is_se:
            self.se = SEModule(2 * exp_channel)
            self.hsig = Hsigmoid()

        self.conv3 = nn.Conv2d(
            in_channels=2 * exp_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.batchn3 = nn.BatchNorm2d(num_features=out_channel)

    def forward(self, x):

        x = self.conv1(x)
        x = self.nl(x)
        x = self.batchn1(x)

        x_1 = self.conv2_a(x)
        x_2 = self.conv2_b(x)

        x_1 = self.batchn2_a(x_1)
        x_2 = self.batchn2_b(x_2)

        x = torch.cat([x_1, x_2], dim=1)

        if self.is_se:
            x = self.se(x)
            x = self.hsig(x)
        x = self.nl(x)

        x = self.conv3(x)

        if self.apply_bn:
            x = self.batchn3(x)

        return x


# Fusenet Model
class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()

        # Defining trainable parameters

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.batchn1 = nn.BatchNorm2d(16)
        self.hswish = Hswish()

        self.fuse1 = Fuse(
            in_channel=16,
            exp_channel=16,
            out_channel=16,
            non_linearity="relu",
            kernel_size=3,
            stride=2,
            is_se=True,
        )
        self.fuse2 = Fuse(
            in_channel=16,
            exp_channel=72,
            out_channel=24,
            non_linearity="relu",
            kernel_size=3,
            stride=2,
            is_se=False,
        )
        self.fuse3 = Fuse(
            in_channel=24,
            exp_channel=88,
            out_channel=24,
            non_linearity="relu",
            kernel_size=3,
            stride=1,
            is_se=False,
        )
        self.fuse4 = Fuse(
            in_channel=24,
            exp_channel=96,
            out_channel=40,
            non_linearity="hswish",
            kernel_size=5,
            stride=2,
            is_se=True,
        )
        self.fuse5 = Fuse(
            in_channel=40,
            exp_channel=240,
            out_channel=40,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )
        self.fuse6 = Fuse(
            in_channel=40,
            exp_channel=240,
            out_channel=40,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )
        self.fuse7 = Fuse(
            in_channel=40,
            exp_channel=120,
            out_channel=48,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )
        self.fuse8 = Fuse(
            in_channel=48,
            exp_channel=144,
            out_channel=48,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )
        self.fuse9 = Fuse(
            in_channel=48,
            exp_channel=288,
            out_channel=96,
            non_linearity="hswish",
            kernel_size=5,
            stride=2,
            is_se=True,
        )
        self.fuse10 = Fuse(
            in_channel=96,
            exp_channel=576,
            out_channel=96,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )
        self.fuse11 = Fuse(
            in_channel=96,
            exp_channel=576,
            out_channel=96,
            non_linearity="hswish",
            kernel_size=5,
            stride=1,
            is_se=True,
        )

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=576, kernel_size=1, stride=1, bias=False
        )
        self.batchn2 = nn.BatchNorm2d(576)

        self.adap = nn.AdaptiveAvgPool2d(1)

        self.conv3 = nn.Conv2d(
            in_channels=576, out_channels=1024, kernel_size=1, stride=1, bias=False
        )

        self.drop = nn.Dropout(p=0.2)

        self.lin = nn.Linear(in_features=1024, out_features=100, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.hswish(x)
        x = self.batchn1(x)

        x = self.fuse1(x)
        x = self.fuse2(x)
        x = self.fuse3(x)
        x = self.fuse4(x)
        x = self.fuse5(x)
        x = self.fuse6(x)
        x = self.fuse7(x)
        x = self.fuse8(x)
        x = self.fuse9(x)
        x = self.fuse10(x)
        x = self.fuse11(x)

        x = self.conv2(x)
        x = self.hswish(x)
        x = self.batchn2(x)
        x = self.adap(x)
        x = self.conv3(x)
        x = self.hswish(x)
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        x = self.lin(x)

        return x


def _initialize_weights(self):

    # weight initialization
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


# Setting GPU Device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    print("No GPU Available!")
    exit()

# Setting up wandb and hyperparameters config
wandb.init(project="fusenet-runs", name="run" + str(run), reinit=True)

config = wandb.config
config.batch_size = 256
config.test_batch_size = 1000
config.epochs = 100
config.lr = 0.0001
config.beta0 = 0.99
config.beta1 = 0.999
config.eps = 1e-08
config.weight_decay = 0

# Setting up training and test dataset

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.test_batch_size, shuffle=False, num_workers=2
)


# Creating instance of Fusenet and uploading to GPU
fusenet = FuseNet().to(device)
fusenet.apply(_initialize_weights)
criterion = nn.CrossEntropyLoss()

# Setting up optimizer with selected hyperparameters
optimizer = optim.Adam(
    fusenet.parameters(),
    lr=config.lr,
    betas=(config.beta0, config.beta1),
    eps=config.eps,
    weight_decay=config.weight_decay,
)

# Linking Fusenet to wandb
wandb.watch(fusenet)

for epoch in range(config.epochs):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = fusenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    wandb.log({"Loss": running_loss})
    correct = 0
    total = 0
    temp = 0

    print("Epoch :", epoch)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = fusenet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()

    wandb.log({"Accuracy": 100 * correct / total})

print("Finished Training")

# Saving model to wandb
torch.save(fusenet.state_dict(), "model" + str(run) + ".h5")
wandb.save("model" + str(run) + ".h5")

run += 1
open("runs.txt", "w").write(str(run))

