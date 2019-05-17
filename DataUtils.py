from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

# Data Preprocess and Data Augumentation
def DataPreprocess1(batch_size):#LeNet
    train_tansform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])

    # Generate Dataset
    mnist_train = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=train_tansform)
    mnist_test = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=test_transform)
    loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(mnist_test, batch_size=batch_size)

    return loader_train,loader_test