import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Losses():
  def __init__(self):
    self.MSELoss=nn.MSELoss()

  def GDLoss(self,reconstruction,batch):
    intersection=torch.sum(reconstruction*batch,dim=(1,2,3))
    cardinality=torch.sum(reconstruction+batch,dim=(1,2,3))

    dice_score=2.*intersection/cardinality
    return torch.mean(1.-dice_score)

  def get_contributes(self,reconstruction,batch):
    contributes={}
    contributes["GDLoss"]=self.GDLoss(reconstruction,batch).item()
    contributes["MSELoss"]=self.MSELoss(reconstruction,batch).item()
    contributes["Total"]=contributes["GDLoss"]+contributes["MSELoss"]
    return contributes

  def __call__(self,reconstruction,batch,epoch):
    loss=self.MSELoss(reconstruction,batch) + self.GDLoss(reconstruction,batch)
    if(epoch<10):
      loss+=self.GDLoss(reconstruction[:,1:],batch[:,1:])
    return loss

class AE(nn.Module):
  def __init__(self, latent_size=100):
    super().__init__()
    self.init_layers(latent_size)
    self.apply(self.weight_init)
    self.loss_function = Losses()

  def init_layers(self,latent_size):
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=4,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=128),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.Conv2d(in_channels=32,out_channels=latent_size,kernel_size=4,stride=2,padding=1)#4,2,1
    )

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=latent_size,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=128),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.LeakyReLU(.2),
      nn.Dropout(0.5),

      nn.ConvTranspose2d(in_channels=32,out_channels=4,kernel_size=4,stride=2,padding=1),
      nn.Softmax(dim=1)
    )

  def weight_init(self,m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
      nn.init.kaiming_uniform_(m.weight)

  def forward(self, x):
    latent = self.encoder(x)
    reconstruction = self.decoder(latent)
    return reconstruction

  def evaluation_routine(self, val_loader):
    epoch_loss={}
    batch_losses={}
    for [batch] in val_loader:
      batch=batch.to(device)
      reconstruction=self.forward(batch)
      for k,v in self.loss_function.get_contributes(reconstruction,batch).items():
        if k not in batch_losses.keys():
          batch_losses[k]=[]
        batch_losses[k].append(v)
    for k in batch_losses.keys():
      epoch_loss[k] = sum(batch_losses[k])/len(batch_losses[k])
    return epoch_loss

  def epoch_end(self, epoch, result):
    output="Epoch [{}], ".format(epoch)
    for k,v in result.items():
      output+="{}: {:.4f} ".format(k,v)
    print(output)

  def training_routine(self, epochs, train_loader, val_loader):
    history = []
    optimizer = torch.optim.Adam(self.parameters(),lr=2e-4,weight_decay=1e-5)
    best_acc = None
    for epoch in range(epochs):
      self.train()
      for [batch] in train_loader:
          batch=batch.to(device)
          optimizer.zero_grad()
          reconstruction=self.forward(batch)
          loss=self.loss_function(reconstruction,batch,epoch)
          loss.backward()
          optimizer.step()
      self.eval()
      result = self.evaluation_routine(val_loader)
      if(best_acc==None or result['Total']<best_acc):
        best_acc=result['Total']
        torch.save(self,"best_ae.pth")
      self.epoch_end(epoch, result)
      history.append(result)
    return history

def plot_history(history):
  losses = [x['Total'] for x in history]
  plt.plot(losses, '-x', label="loss")
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title('Losses vs. No. of epochs')
  plt.grid()
  plt.show()