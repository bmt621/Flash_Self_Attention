from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from timeit import default_timer as timer
import numpy as np
import torch

class Trainer:
    def __init__(self,model: torch.nn.Module,
                 pad_id: int,
                 train_data: DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 schedular: get_scheduler,
                 save_every: int,
                 checkpoint: dict,
                 checkpoint_dir:str,
                 loss_fn: None,
                 gpu_id: torch.device,
                 )->None:
        
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.loss_fn = loss_fn
        self.pad_id = pad_id
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = schedular
    
        
    def _run_batch(self,batch):

        self.optimizer.zero_grad()
        input_id = batch['input_ids'].to(self.gpu_id)
        label = batch['label'].to(self.gpu_id)

        output = self.model(input_id,self.pad_id)
        loss = self.loss_fn(output,label)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _run_epoch(self,epoch):
        losses = []
        for batch in self.train_data:
            loss = self._run_batch(batch)
            losses.append(loss)
        
        return np.mean(losses)
        

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            start_time=timer()
            train_loss = self._run_epoch(epoch)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


    def _save_checkpoint(self,epoch):
        torch.save(self.checkpoint,self.checkpoint_dir)
        print("Saved checkpoint {} at Epoch {}".format(self.checkpoint_dir,epoch))


    
    