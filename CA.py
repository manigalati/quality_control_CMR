import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from medpy.metric import binary

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_layers(kwargs["latent_size"])
        self.apply(self.weight_init)
        self.loss_function = self.Loss(kwargs["functions"], kwargs["settling_epochs"])
        self.metrics = self.Metrics()
        self.optimizer = kwargs["optimizer"](
            self.parameters(),
            lr=kwargs["lr"],
            **{k:v for k,v in kwargs.items() if k in ["weight_decay", "momentum"]}
        )

    def init_layers(self, latent_size):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=latent_size, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    class Loss():
        def __init__(self, functions, settling_epochs):
            self.MSELoss = self.MSELoss()
            self.BKMSELoss = self.BKMSELoss()
            self.BKGDLoss = self.BKGDLoss()
            self.GDLoss = self.GDLoss()
            self.functions = functions
            self.settling_epochs = settling_epochs

        class BKMSELoss:
            def __init__(self):
                self.MSELoss = nn.MSELoss()
            def __call__(self, prediction, target):
                return self.MSELoss(prediction[:,1:], target[:,1:])

        class MSELoss:
            def __init__(self):
                self.MSELoss = nn.MSELoss()
            def __call__(self, prediction, target):
                return self.MSELoss(prediction, target)

        class BKGDLoss:
            def __call__(self, prediction, target):
                intersection = torch.sum(prediction * target, dim=(1,2,3))
                cardinality = torch.sum(prediction + target, dim=(1,2,3))
                dice_score = 2. * intersection / (cardinality + 1e-6)
                return torch.mean(1 - dice_score)
      
        class GDLoss:
            def __call__(self, x, y):
                tp = torch.sum(x * y, dim=(0,2,3))
                fp = torch.sum(x * (1-y), dim=(0,2,3))
                fn = torch.sum((1-x) * y, dim=(0,2,3))
                nominator = 2*tp + 1e-06
                denominator = 2*tp + fp + fn + 1e-06
                dice_score =- (nominator / (denominator+1e-6))[1:].mean()
                return dice_score

        def __call__(self, prediction, target, epoch, validation=False):
            contributes = {f: self.__dict__[f](prediction, target) for f in self.functions}
            if epoch < self.settling_epochs:
                if "BKMSELoss" in contributes:
                    contributes["BKMSELoss"] = self.BKMSELoss(prediction[:,1:], target[:,1:])#TODO: like this impossible old configuration
                if "BKGDLoss" in contributes:
                    contributes["BKGDLoss"] = self.BKGDLoss(prediction[:,1:], target[:,1:])
            contributes["Total"] = sum(contributes.values())
            if validation:
                return {k: v.item() for k,v in contributes.items()}
            else:
                return contributes["Total"]

    class Metrics():
        def __init__(self):
            self.DC = self.DC()
            self.HD = self.HD()

        class DC:
            def __call__(self, prediction, target):
                try:
                    return binary.dc(prediction, target)
                except Exception:
                    return 0

        class HD:
            def __call__(self, prediction, target):
                try:
                    return binary.hd(prediction, target)
                except Exception:
                    return np.nan

        def __call__(self, prediction, target, validation=False):
            metrics = {}
            for c,key in enumerate(["BK_", "RV_", "MYO_", "LV_"]):
                ref = np.copy(target)
                pred = np.copy(prediction)

                ref = np.where(ref!=c, 0, 1)
                pred = np.where(pred!=c , 0, 1)

                metrics[key + "dc"] = self.DC(pred, ref)
                metrics[key + "hd"] = self.HD(pred, ref)
            return metrics

    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        if ckpt_folder is not None and not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        history = []
        best_acc = None
        for epoch in epochs:
            self.train()
            for patient in train_loader:
                for batch in patient:
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    reconstruction = self.forward(batch)
                    loss = self.loss_function(reconstruction, batch, epoch)
                    loss.backward()
                    self.optimizer.step()
                    
            self.eval()
            with torch.no_grad():
                result = self.evaluation_routine(val_loader, epoch)
            
            if ckpt_folder is not None and (best_acc is None or result['Total'] < best_acc or epoch%10 == 0):
                ckpt = os.path.join(ckpt_folder,"{:03d}.pth".format(epoch))
                if best_acc is None or result['Total'] < best_acc:
                    best_acc = result['Total']
                    ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save({"AE": self.state_dict(), "AE_optim": self.optimizer.state_dict(), "epoch": epoch}, ckpt)

            self.epoch_end(epoch, result)
            history.append(result)
        return history

    def evaluation_routine(self, val_loader, epoch):
        epoch_summary = {}
        for patient in val_loader:
            gt, reconstruction = [], []
            for batch in patient:
                batch = {"gt": batch.to(device)}
                batch["reconstruction"] = self.forward(batch["gt"])
                gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt) > 0 else batch["gt"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction) > 0 else batch["reconstruction"]
                for k,v in self.loss_function(batch["reconstruction"], batch["gt"], epoch, validation=True).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k]=[]
                    epoch_summary[k].append(v)
            
            gt = np.argmax(gt.cpu().numpy(), axis=1)
            gt = {"ED": gt[:len(gt)//2], "ES":gt[len(gt)//2:]}
            reconstruction = np.argmax(reconstruction.cpu().numpy(), axis=1)
            reconstruction = {"ED": reconstruction[:len(reconstruction)//2], "ES": reconstruction[len(reconstruction)//2:]}
            for phase in ["ED", "ES"]:
                for k,v in self.metrics(reconstruction[phase], gt[phase]).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k] = []
                    epoch_summary[k].append(v)
        epoch_summary = {k: np.mean(v) for k,v in epoch_summary.items()}
        return epoch_summary

    def epoch_end(self,epoch,result):
        print("\033[1mEpoch [{}]\033[0m".format(epoch))
        header, row = "", ""
        for k,v in result.items():
          header += "{:.6}\t".format(k)
          row += "{:.6}\t".format("{:.4f}".format(v))
        print(header)
        print(row)
    
def plot_history(history):
    losses = [x['Total'] for x in history]
    plt.plot(losses, '-x', label="loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()

#######################
#Hyperparameter Tuning#
#######################

def get_sets(parameters, set_parameters=None):
    if set_parameters is None:
        set_parameters = {k: None for k in parameters.keys()}
    if None not in set_parameters.values():
        yield set_parameters
    else:
        current_index = list(set_parameters.values()).index(None)
        current_parameter = list(set_parameters.keys())[current_index]
        for value in parameters[current_parameter]:
            set_parameters[current_parameter] = value
            loader = get_sets(parameters, set_parameters=set_parameters.copy())
            while True:
                try:
                    yield next(loader)
                except StopIteration:
                    break 

def satisfies_rules(rules, set_parameters):
    for rule in rules:
        keys = np.unique(rule.split('"')[1::2])
        for key in keys:
            if key in set_parameters:
                rule = rule.replace('"' + key + '"', 'set_parameters["' + key + '"]')
        if not eval(rule):
            return False
    return True



def hyperparameter_tuning(parameters, train_loader, val_loader, transform, transform_augmentation, rules=[], fast=False):
    best_dc = 0
    optimal_parameters = None
    for set_parameters in get_sets(parameters):
        if not satisfies_rules(rules, set_parameters):
            continue
        print(set_parameters)

        BATCH_SIZE = set_parameters["BATCH_SIZE"]
        DA = set_parameters["DA"]
        train_loader.set_batch_size(BATCH_SIZE)
        val_loader.set_batch_size(BATCH_SIZE)
        train_loader.set_transform(transform_augmentation if DA else transform)
        val_loader.set_transform(transform)

        ae = AE(**set_parameters).to(device)
        
        history = ae.training_routine(
            range(0,set_parameters["tuning_epochs"]),
            train_loader,
            val_loader
        )
        history = {k:[x[k] for x in history] for k in history[0].keys() if k in ["LV_dc", "MYO_dc", "RV_dc"]}
        history = pd.DataFrame.from_dict(history)
        
        wasBlack = any(np.all(history.values==0, axis=1))
        isNotBlack = all(history.values[-1] > 0.01)
        avg_dc = np.mean(history.values[-1])

        if wasBlack and isNotBlack:
            if avg_dc > best_dc:
                best_dc = avg_dc
                optimal_parameters = set_parameters.copy()
            if fast:
                break

    return optimal_parameters