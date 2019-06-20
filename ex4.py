import torch
import torch.nn.functional as F
from gcommand_loader import GCommandLoader
import torch.optim as optim
from Network import Network


def train(model, train_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)     # CUDA

            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + update
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()


def validate(model, valid_loader, device):
    model.eval()

    correct = 0
    total = 0
    for data in valid_loader:

        samples, labels = data[0].to(device), data[1].to(device)    # CUDA

        outputs = model(samples)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the valid set: %d %%' % (100 * correct / total))


def test_session(model, test_loader, device, file_name):
    with open(file_name, 'w+') as f:
        model.eval()
        for k, (inputs, labels) in enumerate(test_loader):
            i = 0
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            for prediction in preds:
                path = test_loader.dataset.spects[k * preds.shape[0] + i][0]
                splitted = path.split('\\')

                f.write(splitted[len(splitted) - 1] + ', ' + str(prediction.item()) + '\n')
                i += 1
    f.close()


def load_set(path, batch_size=100, num_workers=20, shuffle=True):

    dset = GCommandLoader(path)
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, sampler=None)
    return dloader


if __name__ == '__main__':

    # run on GPU if can do
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 100
    num_workers = 20
    epochs = 15

    # loading the train and test sets
    train_loader = load_set('./data/train')
    test_loader = load_set('./data/test', shuffle=False)

    # init the model and the Adam optimizer
    net = Network()
    net.to(device)  # CUDA
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # train the network, test it and print results to 'test_y'
    train(net, train_loader, optimizer, device, epochs=epochs)
    test_session(net, test_loader, device, 'test_y')

