from tqdm import tqdm
import torch



def train(model, 
          train_data, 
          val_data, 
          device, 
          optimizer,
          scheduler,
          loss_fn, 
          n_epoch,
          save_dir):
    
    n_data = len(train_data.dataset)
    tr_h, val_h = History('min'), History('min')
    
    for epoch in range(1, n_epoch + 1):
        tr_loss = 0.
        model.train()
        for x, y in tqdm(train_data, 'Train', leave=False):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            output = model(x)
            batch_loss = loss_fn(output, y)
            batch_loss.backward()
            optimizer.step()
            tr_loss += batch_loss * y.numel()

        tr_loss /= n_data
        tr_h.add(tr_loss)
        val_loss = evaluate(model, val_data, device, loss_fn)
        val_h.add(val_loss)

        if val_h.better:
            torch.save(model, f'{save_dir}/md.pt')

        scheduler.step(val_loss)

        print(f'Epoch: {epoch:>2} |  '
              f'tr loss: {tr_loss:.4f}  min tr loss: {tr_h.best:.4f}  '
              f'val loss: {val_loss:.4f}  min val loss: {val_h.best:.4f}')

    return torch.load(f'{save_dir}/md.pt', device).eval()


def evaluate(model, val_data, device, loss_fn):
    n_data = len(val_data.dataset)
    val_loss = 0.

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_data, 'Val', leave=False):
            x, y = x.to(device), y.to(device)
            output = model(x)
            batch_loss = loss_fn(output, y)
            val_loss += batch_loss * y.numel()
        val_loss /= n_data

    return val_loss


class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        value = value.item()
        
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value)
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True   