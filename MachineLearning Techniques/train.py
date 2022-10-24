import torch
from torch.utils.data import Dataset, DataLoader

class Trainer():
    def __init__(self,epochs,model,optimizer,train_data,val_data):
        self.epochs=epochs
        self.model=model
        self.optimizer=optimizer
        self.train_loader=DataLoader(My_Dataset(train_data), batch_size=4,shuffle=True)
        self.val_loader=DataLoader(My_Dataset(val_data), batch_size=4,shuffle=True)
        self.loss=nn.CrossEntropyLoss()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        for i in range(self.epochs):
            self.model.train()
            for x,y in self.train_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                self.optimizer.zero_grad()
                y_pred=self.model(x)
                loss=self.loss()
                self.loss.backward()
                self.optimizer.step()
            self.model.eval()
            train_loss,train_acc=self.calculate_accuarcy_loss(self.train_loader)
            val_loss,val_acc=self.calculate_accuarcy_loss(self.val_loader)
            print(f"Train loss : {train_loss} Train acc : {train_acc}  Val loss : {val_loss}  Val acc : {val_acc}")


    
    def calculate_accuarcy_loss(self,dataloader):
        total_loss=0
        count=0
        total_match=0
        with torch.no_grad():
            for x,y in dataloader:
                x=x.to(self.device)
                y=y.to(self.device)
                y_pred=self.model(x)
                loss=self.loss()
                total_loss+=loss.item()
                count+=1
                total_match=(y==y_pred.argmax(axis=1)).sum()
        return loss,total_match/count







