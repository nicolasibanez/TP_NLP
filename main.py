import os
import yaml
from pathlib import Path
import argparse
import yaml
import numpy as np

import torch
import wandb
from tqdm import tqdm

from datasets import load_dataset
from dataset import NLI_Dataset, custom_collate_fn
from model import NLIModel, MLPNLIModel, ALL_MLPNLIModel, TTA_Model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.optim as optim
import torch.nn as nn
import datetime

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

from utils import train, test, valid
from checkpoint import Checkpoint

def training():
    # Load configuration from config.yaml file
    print("Loading config.yaml...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize W&B if use_wandb is set to true
    wandb.login(
        key="0c2765726c4f8398712857ce89d863784837e828"
    )
    wandb_run = None
    if config["training"]["use_wandb"]:
        wandb.init(
            project=config["training"]["wandb_project"],
            entity=config["training"]["wandb_entity"],
            config=config,
        )
        wandb_run = wandb.run

    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(config["data"]["name"])
    dataset = dataset.filter(lambda example: example["label"] != config["data"]["filter_label"])

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # If GPT2, add a cls_token and a sep_token
    if "gpt2" in config["model"]["name"]:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})


    def preprocess_data(examples):
        if "gpt2" in config["model"]["name"]:
            premises = ["[CLS] " + example for example in examples['premise']]
            hypotheses = ["[SEP] " + example + "[SEP]" for example in examples['hypothesis']]
        else:
            premises = [example for example in examples['premise']]
            hypotheses = [example for example in examples['hypothesis']]
        # inputs = [' '.join(premise) + tokenizer.sep_token + ' '.join(hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        # return tokenizer(inputs, padding=False)
        return tokenizer(premises, hypotheses, padding=False, truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
    )

    # Create the model
    model_name = config["model"]["name"]
    model = MLPNLIModel(model_name, config["model"]["model_cfg"]) if config["model"]["model_cfg"] else NLIModel(model_name)
    model_config = AutoConfig.from_pretrained(model_name)

    # FP16
    # model = model.half()
    # model = ALL_MLPNLIModel(model_name, config["model"]["model_cfg"]) if config["model"]["model_cfg"] else NLIModel(model_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"ckpt/{config['model']['name']}_{timestamp}.pth"
    if wandb_run:
        wandb.log({"ckpt_name": ckpt_name})
    # Try loading model_state_dict from checkpoint
    if "model_state_dict" in config["model"].keys():
        # try:
        state_dict = torch.load("ckpt/" + config["model"]["model_state_dict"])
        #     # Extract only the relevant parts of the state dictionary
        #     # transformer_state_dict = {k: v for k, v in state_dict.items() if "transformer" in k}
        #     transformer_state_dict = {k[12:]: v for k, v in state_dict.items() if "transformer" in k}
        #     mlp_state_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items() if "classifier" in k}

        #     # Load the state dictionaries into the model
        #     if config["model"]["model_cfg"]:
        #         print("Loading transformer_state_dict from checkpoint")
        #         model.transformer.load_state_dict(transformer_state_dict)
        #         # model.classifier.load_state_dict(mlp_state_dict)
        #         print("Loaded transformer_state_dict from checkpoint")
        #     else:
        model.load_state_dict(state_dict)
        print("Loaded model_state_dict from checkpoint")
    # except Exception as e:
    #     print(f"Failed to load model_state_dict from checkpoint\n{e}")

    # Create the optimizer
    optimizer = getattr(optim, config["optimizer"]["name"])(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
    )

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Set up device and move model to device
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["use_gpu"] else "cpu")
    model.to(device)

    print(f"Model size : {torch.cuda.memory_reserved(0) / 1024**3} GB")


    # Set up data loaders
    print("Setting up data loaders...")
    # train_dataset = NLI_Dataset(tokenized_datasets, split="train", size=1000)
    max_seq_len = model_config.max_position_embeddings
    if config["data"]["augmentation"]:
        train_dataset = NLI_Dataset(tokenized_datasets, tokenizer, max_seq_len, split="train", augmentation=True)
    else:
        train_dataset = NLI_Dataset(tokenized_datasets, tokenizer, max_seq_len, split="train")
    # valid_dataset = NLI_Dataset(tokenized_datasets, split="validation", size=1000)
    valid_dataset = NLI_Dataset(tokenized_datasets, tokenizer, max_seq_len, split="validation")

    # train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    def collate_fn(batch):
        return custom_collate_fn(batch, tokenizer.pad_token_id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["data"]["batch_size"], shuffle=True, collate_fn=collate_fn)


    # Set up checkpoint
    checkpoint = Checkpoint(higher_is_better=True)

    if config["training"]["use_reduce_lr"]:
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config["training"]["reduce_lr_factor"],
                patience=config["training"]["reduce_lr_patience"],
            )
    else:
        lr_scheduler = None
    # Log the current learning_rate
    if wandb_run:
        # us get_last_lr()
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
    if config["training"]["use_warmup"]:
        total_steps = len(train_dataloader) * config["training"]["num_epochs"]
        warmup_steps = int(total_steps * config["training"]["warmup_proportion"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = None

    # Train the model
    print(f"Training the model for {config['training']['num_epochs']} epochs...")

    model.freeze_transformer()

    frequency_log = 300
    
    # allocated_mem = torch.cuda.memory_allocated(0) / 1024**3  # VRAM allocated in GB
    # reserved_mem = torch.cuda.memory_reserved(0) / 1024**3  # VRAM reserved in GB

    for epoch in range(config["training"]["num_epochs"]):

        if epoch >= config["training"]["unfreeze_epoch"]:
            model.unfreeze_transformer()
        
        train_loss, train_acc = train(
            model,
            device,
            train_dataloader,
            loss_fn,
            optimizer,
            epoch,
            use_wandb=(wandb_run is not None),
            frequency_log=frequency_log,
        )

        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")

        # if wandb_run:
        #     wandb.log({"train_loss": train_loss, "train_acc": train_acc})

        valid_loss, valid_acc = valid(
            model,
            device,
            valid_dataloader,
            loss_fn,
            epoch,
            use_wandb=(wandb_run is not None),
            frequency_log=frequency_log,
        )

        print(f"Epoch: {epoch + 1}, Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")

        if wandb_run:
            wandb.log({"valid_loss": valid_loss, "valid_acc": valid_acc})

        if scheduler is not None:
            scheduler.step()
        if lr_scheduler is not None:
            lr_scheduler.step(train_loss)
        if (scheduler is not None or lr_scheduler is not None) and wandb_run:
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})


        if checkpoint.update(valid_acc):
            print("Saving the model...")
            torch.save(model.state_dict(), ckpt_name)

def testing():
    print("Loading config.yaml...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    

    print("Loading dataset...")
    dataset = load_dataset(config["data"]["name"])
    dataset = dataset.filter(lambda example: example["label"] != config["data"]["filter_label"])

    print("Tokenizing dataset...")
    ckpt = "ckpt/" + config["model"]["model_state_dict"]
    model_name = ckpt.split("/")[1].split("_")[0]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = None
    if config["model"]["TTA"]:
        print("Using TTA Model")
        model = TTA_Model(model_name, config["model"]["model_cfg"], tokenizer)
    else:
        model = MLPNLIModel(model_name, config["model"]["model_cfg"]) if config["model"]["model_cfg"] else NLIModel(model_name)

    model.load_state_dict(torch.load(ckpt))
    model_config = AutoConfig.from_pretrained(model_name)
    
    def preprocess_data(examples):
        premises = [example for example in examples['premise']]
        hypotheses = [example for example in examples['hypothesis']]
        # inputs = [' '.join(premise) + tokenizer.sep_token + ' '.join(hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        # return tokenizer(inputs, padding=False)
        return tokenizer(premises, hypotheses, padding=False, truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["use_gpu"] else "cpu")
    model.to(device)
    # Create the loss function
    loss_fn = nn.CrossEntropyLoss()

    max_seq_len = model_config.max_position_embeddings
    test_dataset = NLI_Dataset(tokenized_datasets, tokenizer, max_seq_len, split="test")
    def collate_fn(batch):
        return custom_collate_fn(batch, tokenizer.pad_token_id)
    
    test_dataloader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    # inputs, labels = next(iter(test_dataloader))
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # test_outputs = model(inputs)

    # print(test_outputs)

    test_loss, test_acc, array = test(
        model,
        device,
        test_dataloader,
        loss_fn,
        0,
    )

    print(f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}")

    results = {
        "model": model_name,
        "checkpoint": ckpt,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    if config["model"]["TTA"]:
        results["TTA"] = True

    # save
    with open("results/" + ("TTA_" if config["model"]["TTA"] else "") +
               ckpt.split("/")[1].split(".")[0] + ".yaml", "w") as f:
        yaml.dump(results, f)

    # save array (using numpy)
    np.save("results/" + ("TTA_" if config["model"]["TTA"] else "") +
            ckpt.split("/")[1].split(".")[0] + ".npy", array)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    args = parser.parse_args()

    if args.mode == "train":
        training()
    elif args.mode == "test":
        testing()
