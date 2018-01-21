import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from dataset import *

# settings and args
parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-state-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before saving state of the model')
parser.add_argument('--save-sample-interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait before saving sample images')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

img_size = 128

inp_size = img_size*img_size*3
mid_size_1 = 500
mid_size_2 = 20

current_dir = os.path.dirname(os.path.abspath(__file__))

# set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load train data
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = ImagesDataset(
    number_sort=True,
    root_dir=os.path.join(current_dir, 'train_imgs'),
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
  train_dataset,
  batch_size=args.batch_size,
  shuffle=True,
  **kwargs
)

# VAE model class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(inp_size, mid_size_1)
        self.fc21 = nn.Linear(mid_size_1, mid_size_2)
        self.fc22 = nn.Linear(mid_size_1, mid_size_2)
        self.fc3 = nn.Linear(mid_size_2, mid_size_1)
        self.fc4 = nn.Linear(mid_size_1, inp_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, inp_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# create a model
model = VAE()
if args.cuda:
    model.cuda()

# loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, inp_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * inp_size

    return BCE + KLD

# create a optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# test function
def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    sample = Variable(torch.randn(64, mid_size_2))

    if args.cuda:
        sample = sample.cuda()

    # epochs
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)

        # save model
        if epoch % args.save_state_interval == 0:
          torch.save(model.state_dict(), 'vae_results/model_' + str(epoch))

        # save sample image
        if epoch % args.save_sample_interval == 0:
          save_image(model.decode(sample).cpu().data.view(64, 3, img_size, img_size),
                     'vae_results/sample_' + str(epoch) + '.png')