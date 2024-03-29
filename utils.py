import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from dataset import NLI_Dataset

import wandb

from torch.cuda.amp import autocast

def train(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    current_epoch: int,
    subset_ratio: float = 1.0,
    # lr_scheduler: optim.lr_scheduler._LRScheduler = None,
    use_wandb: bool = False,
    frequency_log: int = 10
):
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    train_loss = 0
    train_acc = 0
    train_steps = 0

    batch_size = dataloader.batch_size

    nb_batches = int(len(dataloader) * subset_ratio)
    iter_dataloader = iter(dataloader)

    # progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}")
    # progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}")
    progress_bar = tqdm(range(nb_batches), desc=f"Epoch {current_epoch + 1}")

    # scaler = torch.cuda.amp.GradScaler()



    for batch in progress_bar:

        try:
            inputs, labels = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            inputs, labels = next(iter_dataloader)
            
        # overfit on the first batch
        # inputs, labels = first_batch

        optimizer.zero_grad()
        # FP16
        # input_ids, attention_mask, labels = input_ids.half().to(device), attention_mask.half().to(device), labels.to(device)

        inputs, labels = {k: v.to(device) for k, v in inputs.items()}, labels.to(device)
        outputs = model(inputs) 

        loss = loss_fn(outputs, labels)

        # with torch.cuda.amp.autocast():
        #     outputs = model(input_ids, attention_mask=attention_mask)
        #     loss = loss_fn(outputs, labels)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1

        logits = outputs.detach().cpu()
        label_ids = labels.detach().cpu()
        batch_acc = accuracy_score(label_ids, logits.argmax(dim=1))
        train_acc += batch_acc * len(label_ids)

        # Update the current loss and accuracy at each batch size
        current_loss = train_loss / (batch + 1)
        current_acc = train_acc / ((batch + 1) * batch_size)

        # learning rate scheduler step
        # if lr_scheduler is not None:
        #     lr_scheduler.step(current_loss)
        #     if use_wandb:
        #         wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})


        progress_bar.set_postfix({'Current Loss': f'{current_loss:.3f}', 'Current Acc': f'{current_acc:.3f}'})
        if use_wandb and batch % frequency_log == 0:
            wandb.log({'train_loss': current_loss, 'train_acc': current_acc}, step=int(current_epoch * len(dataloader.dataset) * subset_ratio) + (batch + 1) * batch_size)


    train_loss /= train_steps
    size = nb_batches * batch_size
    train_acc /= size

    return train_loss, train_acc


def valid(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    current_epoch: int,
    use_wandb: bool = False,
    frequency_log: int = 10,
    TTA: bool = False
):
    model.eval()

    test_loss = 0
    test_acc = 0
    test_steps = 0

    batch_size = dataloader.batch_size

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}")
    
    for batch, (inputs, labels) in enumerate(progress_bar):
        # input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        inputs, labels = {k: v.to(device) for k, v in inputs.items()}, labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        test_loss += loss.item()
        test_steps += 1

        logits = outputs.detach().cpu()
        label_ids = labels.detach().cpu()
        batch_acc = accuracy_score(label_ids, logits.argmax(dim=1))
        test_acc += batch_acc * len(label_ids)

        # Update the current loss and accuracy at each batch size
        current_loss = test_loss / (batch + 1)
        current_acc = test_acc / ((batch + 1) * batch_size)
        progress_bar.set_postfix({'Current Loss': f'{current_loss:.3f}', 'Current Acc': f'{current_acc:.3f}'})
        # if use_wandb and batch % frequency_log == 0:
        #     wandb.log({'test_loss': current_loss, 'test_acc': current_acc}, step = current_epoch * len(dataloader.dataset) + (batch + 1) * batch_size)


    test_loss /= test_steps
    test_acc /= len(dataloader.dataset)

    return test_loss, test_acc


def test(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    current_epoch: int,
    use_wandb: bool = False,
    frequency_log: int = 10,
    TTA: bool = False
):
    model.eval()

    test_loss = 0
    test_acc = 0
    test_steps = 0

    batch_size = dataloader.batch_size

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}")
    
    outputs_list = []

    for batch, (inputs, labels) in enumerate(progress_bar):
        # input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        inputs, labels = {k: v.to(device) for k, v in inputs.items()}, labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        outputs_list.append(outputs.detach().cpu().numpy())

        test_loss += loss.item()
        test_steps += 1

        logits = outputs.detach().cpu()
        label_ids = labels.detach().cpu()
        batch_acc = accuracy_score(label_ids, logits.argmax(dim=1))
        test_acc += batch_acc * len(label_ids)

        # Update the current loss and accuracy at each batch size
        current_loss = test_loss / (batch + 1)
        current_acc = test_acc / ((batch + 1) * batch_size)
        progress_bar.set_postfix({'Current Loss': f'{current_loss:.3f}', 'Current Acc': f'{current_acc:.3f}'})
        # if use_wandb and batch % frequency_log == 0:
        #     wandb.log({'test_loss': current_loss, 'test_acc': current_acc}, step = current_epoch * len(dataloader.dataset) + (batch + 1) * batch_size)


    test_loss /= test_steps
    test_acc /= len(dataloader.dataset)

    outputs_list = np.concatenate(outputs_list, axis=0)

    return test_loss, test_acc, outputs_list