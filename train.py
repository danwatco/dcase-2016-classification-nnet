#!/usr/bin/env python3
##################################################
## File to train image classification model, written for Deep Learning coursework 2021
## Designed to be run on a HPC cluster.
##################################################
##################################################
## Author: Dan Watkinson
##################################################

import time
from multiprocessing import Value, cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import argparse
from pathlib import Path
from dataset import DCASE, DCASE_Non_Full
import random

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Train a CNN on DCASE',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
default_dataset_dir = Path.cwd() / "ADL_DCASE_DATA"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of spectrograms within each mini-batch"
)
parser.add_argument(
    "--epochs",
    default=400,
    type=int,
    help="Number of epochs (passes through entire dataset)"
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument("--checkpoint-path", type=Path, )
parser.add_argument("--skip-non-full", type=bool, default=False)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--augmentation", type=bool, default=False)

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    
    ## Setup transforms
    transform = FrequencyMask(F=15, num_masks=1)
    time_transform = TimeMask(T=4, num_masks=1)
    if args.augmentation:
        both = transforms.Compose([transform, time_transform])
    else:
        both = None

    # Read files with train/test splits
    train_fname = Path('train_files.txt')
    test_fname = Path('test_files.txt')

    if train_fname.is_file() and test_fname.is_file():
        with open(train_fname, 'r') as f:
            train_files = [x.strip('\n') for x in f.readlines()]
        with open(test_fname, 'r') as f:
            test_files = [x.strip('\n') for x in f.readlines()]
    else:
        print('No train/test split specified')
        exit()

    # Define model, criterion and optimizer
    model = CNN(height=60, width=150, channels=1, class_count=15, dropout=args.dropout)

    criterion = nn.CrossEntropyLoss() # categorical cross entropy?

    # Adam Optimizer as used in paper 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.skip_non_full == False:
        ## Setup data for non full training - see dataset.py for modifications
        # # Extra param to filter dataset
        train_dataset_non_full = DCASE_Non_Full(Path(args.dataset_root) / 'development', 3, train_files)
        test_dataset_non_full = DCASE_Non_Full(Path(args.dataset_root) / 'development', 3, test_files)


        train_loader = torch.utils.data.DataLoader(
            train_dataset_non_full,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset_non_full,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=False,
            num_workers=args.worker_count
        )
        # logs stuff
        log_dir = get_summary_writer_log_dir(args, full=False)
        print(f"Writing logs to {log_dir}")
        summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
        )
        # Initalise trainer
        trainer = Trainer(
            model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, args.checkpoint_path
        )

        trainer.train(
            args.epochs,
            args.val_frequency,
            print_frequency=args.print_frequency,
            log_frequency=args.log_frequency
        )

        max_acc = trainer.max_accuracy
        print(f'Accuracy max at {max_acc[0]} epochs')
        summary_writer.close()
        exit()
    
    ## Hardcoded for ease of checking success/fail cases
    check = False
    
    training_dataset = DCASE(Path(args.dataset_root) / 'development', 3, transform=transform)    
    eval_dataset = DCASE(Path(args.dataset_root) / 'evaluation', 3)   

    # Code used to check success/fail cases and find which inputs are being missclassified
    if check:
        specs = []
        class_chosen = 8
        for i in range(0, len(eval_dataset)):
            spec, label = eval_dataset[i]
            if label == class_chosen:
                specs.append(i)
        model.load_state_dict(torch.load(Path('./end_checkpoint_1'), map_location=torch.device('cpu')))
        # bus = [9, 103, 114]
        with torch.no_grad():
            model.eval()
            for i in specs:
                spec, label = eval_dataset[i]
                logits = model(spec)
                pred = logits.argmax(dim=1).cpu().numpy()
                print(f'Class: {class_chosen}, Prediction: {pred}')
            exit()

    whole_train_loader = torch.utils.data.DataLoader(
        training_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count
    )

    log_dir = get_summary_writer_log_dir(args, full=True)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )

    full_trainer = Trainer(model, whole_train_loader, eval_loader, criterion, optimizer, summary_writer, DEVICE, args.checkpoint_path)

    full_trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        full=True
    )

    summary_writer.close()
    

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(p=dropout)

        # Convolution 1 128 kernels 5x5
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=128,
            kernel_size=(5,5)
        )
        self.initialise_layer(self.conv1)
        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(5,5), stride=(5,5))

        # Convolution 2 256 kernels 5x5
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=256,
            kernel_size=(5,5)
        )
        self.initialise_layer(self.conv2)
        # Batch Norm
        self.batch2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        # Adaptive Max Pooling
        self.pool2 = nn.AdaptiveMaxPool2d((4, 1))

        # FC Layers
        self.fc1 = nn.Linear(256 * 4 * 1, 15)
        self.initialise_layer(self.fc1)

    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # reshape to create correct size for layers
        x = images.view(-1, 1, 60, 150)

        # Conv 1
        x = F.relu(self.batch1(self.conv1(x)))
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        
        # Conv 2
        x = F.relu(self.batch2(self.conv2(x)))
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())

        # Flatten for fc layer
        x = torch.flatten(x, 1)
        # Dropout to experiment with
        x = self.dropout(x)
        x = self.fc1(x)

        # Returning correct data size
        x = x.view(-1, 10, 15)

        # Aveeage oer all clips to get prediction
        return x.mean(1)
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        checkpoint_path: Path
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.checkpoint_path = checkpoint_path
        self.max_accuracy = (0,0) # epoch, accuracy

    def train(self, epochs: int, val_frequency: int, print_frequency: int = 20, log_frequency: int = 5, start_epoch: int = 0, full: bool = False, checkpoint = None):
            # if checkpoint is not None:
            #     self.model.load_state_dict(torch.load(checkpoint))
            #     exit()

            self.model.train()
            for epoch in range(start_epoch, epochs):
                self.model.train()
                data_load_start_time = time.time()
                for batch, labels in self.train_loader:
                    # Transfer data to gpu
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    data_load_end_time = time.time()
                    # Model forward pass
                    logits = self.model.forward(batch)
                    # Calculate loss given forward pass
                    loss = self.criterion(logits, labels)
                    # Backward pass
                    loss.backward()
                    # Optimizer steps
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    with torch.no_grad():
                        preds = logits.detach().cpu().argmax(-1)
                        accuracy = compute_accuracy(labels.cpu(), preds)
                    data_load_time = data_load_end_time - data_load_start_time
                    step_time = time.time() - data_load_end_time
                    if ((self.step + 1) % log_frequency) == 0:
                        # log metrics
                        self.log_metrics(epoch, accuracy, loss.detach().cpu(), data_load_time, step_time)
                    if ((self.step + 1) % print_frequency) == 0:
                        # print metrics
                        self.print_metrics(epoch, accuracy, loss.detach().cpu(), data_load_time, step_time)
                    self.step += 1
                    data_load_start_time = time.time()
                self.summary_writer.add_scalar("epoch", epoch, self.step)
                if ((epoch + 1) % val_frequency) == 0:
                    end = self.validate(epoch, full)
                    # validate will enter validation mode so switch back to train
                    self.model.train()
                    
                    if end:
                        break
                    
            if full is False:
                print(f'Best Accuracy at epoch: {self.max_accuracy[0]}, accuracy: {self.max_accuracy[1] * 100:2.2f}')
            else:
                self.createConfustionMatrix()
                torch.save(self.model.state_dict(), Path('end_checkpoint'))
    
    @torch.no_grad()
    def createConfustionMatrix(self):
            with torch.no_grad():
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])
                for batch, labels in self.val_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device).cpu()
                    logits = self.model(batch)
                    preds = logits.argmax(dim=1).cpu()
                    # print(preds)
                    all_preds = torch.cat((all_preds, preds.float()), dim=0)
                    all_labels = torch.cat((all_labels, labels.float()), dim=0)

                display_labels = ['beach',  "bus", "cafe/restaurant", "car", "city center", "forest path", "grocery store", 
                "home", "library", "metro station", "office", "park", "residential area", "train", "tram"]
                # create confusion matrix with sklearn
                cf_matrix = confusion_matrix(all_labels.numpy(), all_preds.numpy())
                # print(cf_matrix)
                df_cm = pd.DataFrame(cf_matrix, index = [i for i in display_labels],
                     columns = [i for i in display_labels])
                # plt.figure(figsize = (12,7))
                f, ax = plt.subplots(figsize=(12,12))
                sn.heatmap(df_cm, annot=True, cmap='Greens', cbar=False, ax=ax)

                ax.set_xlabel('Predicted Class')
                ax.set_ylabel('True Class')
                ax.tick_params(axis='x', labelrotation=-45)

                f.savefig('confusion.png')


    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy",
            {"train" : accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self, epoch, full: bool) -> bool:
        end = False

        results = {"preds" : [], "labels" : []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(np.array(results["labels"]), np.array(results["preds"]))
        average_loss = total_loss / len(self.val_loader)

        if full is False:
            if accuracy > self.max_accuracy[1]:
                # save params
                self.max_accuracy = (epoch, accuracy)
                print(f'Saving Model to {self.checkpoint_path}')
                torch.save(self.model.state_dict(), self.checkpoint_path)
            elif (epoch - self.max_accuracy[0]) > 99 and epoch > 200:
                end = True

        self.summary_writer.add_scalars(
            "accuracy",
            {"test" : accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"test" : average_loss},
            self.step
        )

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        # if accuracy > 0.86:
        #     torch.save(self.model.state_dict(), Path('./high_accuracy'))
        #     if epoch > 99:
        #         end = True

        return end

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def get_summary_writer_log_dir(args: argparse.Namespace, full) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    # tb_log_dir_prefix = f'CNN__bn_bs={args.batch_size}_lr={args.learning_rate}_momentum=0.9_run_'
    tb_log_dir_prefix = (
      f"CNN_ASC_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"dropout={args.dropout}_"
      f"full={full}_"
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


## Code adapted from that developed by Zach Caceres and Jenny Cai see https://github.com/zcaceres/spec_augment/

class FrequencyMask(object):
    def __init__(self, F: int, num_masks: int):
        self.F = F
        self.num_masks = num_masks

    def __call__(self, sample):
        cloned = sample.clone()
        num_mel_channels = cloned.shape[1]
        # print(num_mel_channels)
    
        for i in range(0, self.num_masks):        
            f = random.randrange(0, self.F)
            f_zero = random.randrange(0, num_mel_channels - f)
            # print(f, f_zero)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            cloned[:,f_zero:mask_end,:] = cloned.mean()
        return cloned

class TimeMask(object):
    def __init__(self, T:int, num_masks:int):
        self.T = T
        self.num_masks = num_masks

    def __call__(self, sample):
        cloned = sample.clone()
        len_spectro = cloned.shape[2]

        for i in range(0, self.num_masks):
            t = random.randrange(0, self.T)
            t_zero = random.randrange(0, len_spectro - t)
            # print(t, t_zero)
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            cloned[:,:,t_zero:mask_end] = cloned.mean()
        return cloned  

if __name__ == "__main__":
    main(parser.parse_args())
