import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import os
import struct
import lmdb
import datetime

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (roc_curve, confusion_matrix, classification_report)

LMDB_DIR = "./lmdb"
SAVE_DIR = "."

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stopping_patience": 14,
    "epochs": 100,
    "view_len": 150_000,
    "pad_token": 256,
    "accumulation_steps": 2
}

PIN = torch.cuda.is_available()

class ByteDataset(Dataset):
    def __init__(self, lmdb_path, train=False):
        self.lmdb_path = str(lmdb_path)
        self.train     = train

        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin(write = False) as txn:
            self.length = int(txn.get(b"num-samples").decode())
        env.close()

        self.env = None
        self.txn = None

    def _init_worker(self):
        self.env = lmdb.open(self.lmdb_path, 
                             readonly = True, 
                             lock = False, 
                             readahead = True)
        self.txn = self.env.begin(write = False)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.txn is None:
            self._init_worker()

        key_data  = f"data-{idx}".encode()
        key_label = f"label-{idx}".encode()
        key_len   = f"len-{idx}".encode()

        raw_buf = self.txn.get(key_data)
        if raw_buf is None:
            raise KeyError(f"missing key {key_data!r}")
        
        label = int(self.txn.get(key_label))
        len_buf = self.txn.get(key_len)

        if len_buf is None:
            print(f"len_buf is none")
            L = len(raw_buf)
        else:
            L = struct.unpack("<I", len_buf)[0]

        if L >= CONFIG["view_len"]:
            start = random.randint(0, L - CONFIG["view_len"]) if self.train else (L - CONFIG["view_len"]) // 2
            window = raw_buf[start : start + CONFIG["view_len"]]
            x = np.frombuffer(window, dtype = np.uint8).astype(np.int64)
        else:
            x = np.full(CONFIG["view_len"], CONFIG["pad_token"], dtype = np.int64)
            x[:L] = np.frombuffer(raw_buf[:L], dtype = np.uint8)

        X = torch.from_numpy(x)
        return X, label



class CNN(torch.nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.device = CONFIG["device"]
        self.loss_function = loss_function
        self.train_loss = []
        self.validation_loss = []

        self.embedding = torch.nn.Embedding(
            num_embeddings = CONFIG["pad_token"] + 1,
            embedding_dim  = 64,
            padding_idx = CONFIG["pad_token"]
        )

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size = 7, stride = 2, padding = 3, dilation = 1),
            torch.nn.GroupNorm(8, 128),
            torch.nn.GELU(),
            torch.nn.Dropout1d(p = 0.15),
            torch.nn.MaxPool1d(kernel_size = 2, stride = 2),

            torch.nn.Conv1d(128, 256, kernel_size = 7, stride = 2, padding = 3, dilation = 1),
            torch.nn.GroupNorm(8, 256),
            torch.nn.GELU(),
            torch.nn.Dropout1d(p = 0.20),

            torch.nn.Conv1d(256, 512, kernel_size = 7, stride = 1, padding = 6, dilation = 2),
            torch.nn.GroupNorm(8, 512),
            torch.nn.GELU(),
            torch.nn.Dropout1d(p = 0.20),
            torch.nn.MaxPool1d(kernel_size = 2, stride = 2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 2, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p = 0.35),

            torch.nn.Linear(256, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(p = 0.15),

            torch.nn.Linear(32, 1)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, view_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, view_len]
        x = self.features(x)   # [batch_size, channels, reduced_len]

        x_avg = x.mean(dim=-1)  # [batch_size, channels]
        x_max = x.amax(dim=-1)  # [batch_size, channels]
        x = torch.cat([x_avg, x_max], dim=1)  # [batch_size, channels*2]

        x = self.fc(x)  # [batch_size, 1]

        return x


def create_dataloader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size = CONFIG["batch_size"],
        shuffle = shuffle,
        num_workers = 4,
        pin_memory = PIN,
        persistent_workers = True,
        worker_init_fn = lambda worker_id: dataset._init_worker()
    )

class Metrics:
    @staticmethod
    def plot_loss(train_loss, validation_loss):
        plt.plot(train_loss, label="Train")
        plt.plot(validation_loss, label="Validation")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi = 150)
        plt.close()



def find_threshold(model, dataset):
    model.eval()
    dataloader = create_dataloader(dataset)

    y_true = []
    y_pred_prob = []
    with torch.no_grad():
        for images, labels in dataloader:
            y_true.extend(labels.cpu().numpy().tolist())

            images = images.to(model.device)
            logits = model(images) # [batch_size, 1]
            probs = torch.sigmoid(logits).squeeze(1) # [batch_size]
            y_pred_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    J = tpr - fpr
    best_idx = np.argmax(J)
    optimal_threshold = thresholds[best_idx]
    return optimal_threshold

def evaluate_model(model, dataset, threshold = 0.50, draw_report = False):
    model.eval()
    dataloader = create_dataloader(dataset)

    val_loss = 0.0
    corrects = 0
    N = 0

    y_true = []
    y_scores = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            # labels = [batch_size]
            y_true.extend(labels.cpu().numpy().tolist())

            images = images.to(model.device)
            labels = labels.to(model.device)

            logits = model(images) # [batch_size, 1]
            probs = torch.sigmoid(logits).squeeze(1) # [batch_size]
            loss = model.loss_function(logits, labels.float().unsqueeze(1))

            val_loss += loss.item() * images.size(0)
            N += images.size(0)
            y_scores.extend(probs.cpu().numpy())

    y_pred = (np.array(y_scores) > threshold).astype(int)
    corrects = (y_pred == np.array(y_true)).sum()

    if draw_report:
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))

    return (val_loss / N), (corrects / N)


def train_model(model, train_dataset, validation_dataset):
    train_loader = create_dataloader(train_dataset, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = CONFIG["learning_rate"],
                                  weight_decay = CONFIG["weight_decay"]
                                 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience  = 4,
        factor    = 0.5,
        min_lr    = 1e-6,
        threshold = 1e-3
    )

    best_validation_loss = float("inf")
    patience = CONFIG["early_stopping_patience"]
    counter = 0
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"==== EPOCH {epoch} ====")
        print(f"Start: {datetime.datetime.now().time()}")
        model.train()
        total_loss = 0
        N = 0
        corrects = 0
        
        optimizer.zero_grad(set_to_none=True)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            with torch.amp.autocast("cuda"):
                logits = model(images) # [batch_size, 1]
                loss_unscaled = model.loss_function(logits, labels.float().unsqueeze(1))
                loss = loss_unscaled / CONFIG["accumulation_steps"]

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG["accumulation_steps"] == 0 or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                

            predicted = torch.sigmoid(logits.squeeze(1)) > 0.50 # [batch_size]
            corrects += (predicted == labels).sum().item() # [batch_size]

            total_loss += loss_unscaled.item() * images.size(0)
            N += images.size(0)

        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_dataset,
            draw_report = (epoch % 5 == 0)
        )
        
        scheduler.step(validation_loss)

        train_loss = total_loss / N
        train_accuracy = corrects / N

        model.train_loss.append(train_loss)
        model.validation_loss.append(validation_loss)

        print(f"Learning rate = {optimizer.param_groups[0]['lr']}")
        print(f"Train accuracy = {(train_accuracy) * 100:.4f}% | Train loss = {model.train_loss[-1]:.5f}")
        print(f"Validation accuracy = {(validation_accuracy * 100):.4f}% | Validation loss = {validation_loss:.5f}")
        print(f"End: {datetime.datetime.now().time()}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_validation_loss': best_validation_loss,
                'early_stop_counter': counter,
            }, os.path.join(SAVE_DIR, "model.pth"))

            print(f"Saved model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    Metrics.plot_loss(model.train_loss, model.validation_loss)

def evaluate_multicrop_seq(model, dataset, threshold=0.5, num_crops=7):
    model.eval()
    base_loader = create_dataloader(dataset)                 # center crop pass
    rand_loader = create_dataloader(ByteDataset(dataset.lmdb_path, train=True))

    import numpy as np
    y_true, score_sum = [], np.zeros(len(dataset), dtype=np.float64)
    with torch.no_grad():
        # pass 1: center crop
        idx = 0
        for x, y in base_loader:
            y_true.extend(y.numpy().tolist())
            p = torch.sigmoid(model(x.to(model.device))).squeeze(1).cpu().numpy()
            n = len(p); score_sum[idx:idx+n] += p; idx += n
        # passes 2..K: random crops
        for _ in range(num_crops - 1):
            idx = 0
            for x, _ in rand_loader:
                p = torch.sigmoid(model(x.to(model.device))).squeeze(1).cpu().numpy()
                n = len(p); score_sum[idx:idx+n] += p; idx += n

    scores = score_sum / num_crops
    y_pred = (scores > threshold).astype(int)
    acc = (y_pred == np.array(y_true)).mean()
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Accuracy after {num_crops} crops: {acc*100:.4f}%")
    return acc


print("Creating train_dataset")
train_dataset = ByteDataset(lmdb_path=os.path.join(LMDB_DIR, "train.lmdb"), train = True)
print("Finished")

print("Creating validation_dataset")
validation_dataset = ByteDataset(lmdb_path=os.path.join(LMDB_DIR, "validation.lmdb"))
print("Finished")

test_dataset = ByteDataset(lmdb_path=os.path.join(LMDB_DIR, "test.lmdb"))

loss_fn = torch.nn.BCEWithLogitsLoss()

cnn = CNN(loss_function = loss_fn)
# train_model(cnn, train_dataset, validation_dataset)


# print("Training finished.")
# print("Computing optimal threshold")
# optimal_threshold = find_threshold(cnn, validation_dataset)
# print(f"Validation dataset evaluation using threshold {optimal_threshold}:")
# evaluate_model(cnn, validation_dataset, threshold = optimal_threshold, draw_report = True)

# print("Evaluate multicrop seq")
# evaluate_multicrop_seq(cnn, validation_dataset, optimal_threshold)


checkpoint = torch.load("model.pth", map_location=CONFIG["device"])
cnn.load_state_dict(checkpoint["model_state_dict"])
optimal_threshold = 0.42
evaluate_model(cnn, test_dataset, threshold = optimal_threshold, draw_report = True)
print("Evaluate multicrop seq")
evaluate_multicrop_seq(cnn, test_dataset, optimal_threshold)
