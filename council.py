import os
import pefile
import joblib
import pandas as pd
import torch
import numpy as np
import re
import time

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def entropy(chunk):
    arr = np.frombuffer(chunk, dtype = np.uint8)
    counts = np.bincount(arr, minlength = 256)
    probs = counts[counts > 0] / len(arr)
    return -np.sum(probs * np.log2(probs))

class PE:
    def __init__(self, path):
        self.path = path
        self.pe = pefile.PE(self.path)
        
        with open(path, "rb") as f:
            self.raw = f.read()



    def get_file_size(self):
        return os.path.getsize(self.path)

    def get_sections(self):
        sections = []
        for section in self.pe.sections:
            sections.append(section)
        
        return sections

    def get_imported_symbols(self):
        out = {}
        ordinal_only = 0
        try:
            for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
                dll = entry.dll.decode(errors = "ignore")
                out[dll] = []

                for imp in entry.imports:
                    if imp.name:
                        out[dll].append(imp.name.decode(errors = "ignore"))
                    else:
                        ordinal_only = 1
        except Exception:
            pass
        
        return out, ordinal_only
    
    
    def get_export_count(self):
        try:
            return len(self.pe.DIRECTORY_ENTRY_EXPORT.symbols)
        except Exception:
            return 0
    
    def get_ascii_strings(self):
        ascii_strings = re.findall(rb"[ -~]{4,}", self.raw)
        return ascii_strings

    def get_unicode_strings(self):
        unicode_strings = re.findall(rb"(?:[\x20-\x7E]\x00){4,}", self.raw)
        return unicode_strings
    
    def get_overlay_info(self):
        file_size = self.get_file_size()
        last_section = max(self.pe.sections, key = lambda s: s.PointerToRawData + s.SizeOfRawData)
        overlay_start = last_section.PointerToRawData + last_section.SizeOfRawData
        overlay_size = max(0, file_size - overlay_start)
        overlay_ratio = overlay_size / file_size if file_size > 0 else 0
        return {
            "has_overlay": int(overlay_size > 0),
            "overlay_ratio": overlay_ratio
        }
    
    def get_header_flags(self):
        oh = self.pe.OPTIONAL_HEADER
        file_size = os.path.getsize(self.path)
        ep_ratio = oh.AddressOfEntryPoint / file_size if file_size > 0 else 0

        return {
            "entry_point_ratio": ep_ratio,
            "dllchars_aslr": int(bool(oh.DllCharacteristics & 0x0040)),
            "dllchars_nx": int(bool(oh.DllCharacteristics & 0x0100))
        }


    def get_byte_hist16(self):
        counts = np.bincount(np.frombuffer(self.raw, dtype = np.uint8), minlength = 256)
        bins16 = counts.reshape(16, 16).sum(axis = 1)
        return (bins16 / max(1, counts.sum())).tolist()

    def get_features(self):
        sections = self.get_sections()
        raw_sizes = []
        entropies = []
        wx_cnt = 0
        
        for section in sections:
            ent = entropy(section.get_data())
            
            entropies.append(ent)
            raw_sizes.append(section.SizeOfRawData)
            chars = section.Characteristics
            if (chars & 0x80000000) and (chars & 0x20000000):  # write & exec
                wx_cnt += 1


        imported_symbols, ordinal_only = self.get_imported_symbols()
        dlls = len(imported_symbols)
        apis = sum(len(v) for v in imported_symbols.values())

        ascii_strings = self.get_ascii_strings()
        unicode_strings = self.get_unicode_strings()
        
        features = {
            "num_sections": len(sections),
            "dlls": np.log1p(dlls),
            "apis": np.log1p(apis),
            "ordinal_only_imports": ordinal_only,
            "sections_entropy_mean": float(np.mean(entropies)),
            "sections_entropy_max": float(np.max(entropies)),
            "sections_entropy_std": float(np.std(entropies)),
            "sections_raw_mean_log": np.log1p(float(np.mean(raw_sizes))),
            "sec_wx_cnt": wx_cnt,
            "ascii_cnt": np.log1p(len(ascii_strings)),
            "unicode_cnt": np.log1p(len(unicode_strings)),
            "export_cnt_log": np.log1p(self.get_export_count()),
            "timestamp_age_days_log": np.log1p(max(0, (int(time.time()) - self.pe.FILE_HEADER.TimeDateStamp) / 86400))
        }

        features.update(self.get_overlay_info())
        features.update(self.get_header_flags())

        hist16 = self.get_byte_hist16()
        for i, v in enumerate(hist16):
            features[f"hist16_{i}"] = v

        return features


CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "view_len": 150_000,
    "pad_token": 256,
}


def prepare_bytes(file_path):
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    view_len = 160_000
    pad_token = 256

    if len(raw_bytes) >= view_len:
        start = (len(raw_bytes) - view_len) // 2
        window = raw_bytes[start:start+view_len]
        x = np.frombuffer(window, dtype=np.uint8).astype(np.int64)
    else:
        x = np.full(view_len, pad_token, dtype=np.int64)
        x[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)

    x_tensor = torch.tensor(x).unsqueeze(0).to(CONFIG["device"])  # [1, view_len]
    return x_tensor



class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.threshold = 0.38
        
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

        checkpoint = torch.load("./models/1d_cnn/model.pth", map_location=CONFIG["device"])
        self.load_state_dict(checkpoint["model_state_dict"])
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.eval()
        self.to(CONFIG["device"])

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, view_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, view_len]
        x = self.features(x)   # [batch_size, channels, reduced_len]

        x_avg = x.mean(dim=-1)  # [batch_size, channels]
        x_max = x.amax(dim=-1)  # [batch_size, channels]
        x = torch.cat([x_avg, x_max], dim=1)  # [batch_size, channels*2]

        x = self.fc(x)  # [batch_size, 1]

        return x
    
    def predict(self, file):
        x = self._prepare_bytes(file)
        with torch.no_grad():
            logit = self(x)
            prob = torch.sigmoid(logit).item()
        return prob

    def _prepare_bytes(self, file_path):
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        view_len = CONFIG["view_len"]
        pad_token = CONFIG["pad_token"]

        if len(raw_bytes) >= view_len:
            start = (len(raw_bytes) - view_len) // 2
            window = raw_bytes[start:start+view_len]
            x = np.frombuffer(window, dtype=np.uint8).astype(np.int64)
        else:
            x = np.full(view_len, pad_token, dtype=np.int64)
            x[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype=np.uint8)

        x_tensor = torch.tensor(x).unsqueeze(0).to(CONFIG["device"])  # [1, view_len]
        return x_tensor

def predict_proba(self, file_path):
    x = self._prepare_bytes(file_path)
    with torch.no_grad():
        logit = self(x)  # call forward pass
        prob = torch.sigmoid(logit).item()
    return prob


def isPE(sample):
    try:
        with open(sample, "rb") as f:
            magic = f.read(2)
        
        return magic == b"MZ"
    except Exception:
        return False

def get_accuracy(models):
    csv = pd.read_csv("./data/test.csv")
    hashes = csv["hash"].tolist()
    labels = csv["label"].tolist()

    correct = 0

    y_true = []
    y_pred = []

    for h, true_label in tqdm(zip(hashes, labels), total=len(hashes), desc="Evaluating Council"):
        try:
            pe_obj = PE(f"./samples/{h}.exe")
            X = pd.DataFrame([pe_obj.get_features()])

            probs = []
            for name, model in models.items():
                if name == "CNN":
                    prob = model.predict(f"./samples/{h}.exe")
                else:
                    prob = model.predict_proba(X)[0][1]
                probs.append(prob)

            avg_prob = sum(probs) / len(probs)
            predicted_label = int(avg_prob > 0.5)

            y_true.append(true_label)
            y_pred.append(predicted_label)

            if predicted_label == true_label:
                correct += 1
            else:
                print(f"For sample {h} council predicted {predicted_label}. Correct answer: {true_label}")

        except Exception as e:
            print(f"Error with {h}: {e}")
            continue
    
    # Compute metrics
    acc = sum([int(p == t) for p, t in zip(y_pred, y_true)]) / len(y_true)
    print(f"\nCouncil accuracy is {acc * 100:.2f}%")

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Optional: print precision/recall/F1
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


files = [f for f in os.listdir() if os.path.isfile(f)]
PEs = [f for f in files if isPE(f)]

models = {
    "k-NN": joblib.load("./models/knn/model.joblib"),
    "Naive Bayes": joblib.load("./models/naive_bayes/model.joblib"),
    "SVM": joblib.load("./models/svm/model.joblib"),
    "Random Forest": joblib.load("./models/random_forest/model.joblib"),
    "XGBoost": joblib.load("./models/xgboost/model.joblib"),
    "CNN": CNN()
}

# get_accuracy(models)

if len(PEs) > 0:
    print("Judges loaded")

    for pe in PEs:
        try:
            pe_obj = PE(pe)
            X = pe_obj.get_features()
            X = pd.DataFrame([X])

            verdicts = []
            for name, model in models.items():
                if name == "CNN":
                    verdicts.append(model.predict(pe))
                else:
                    verdicts.append(model.predict_proba(X)[0][1])
                print(f"Judge {name}: {verdicts[-1] * 100:.2f}%")
            
            s = sum(verdicts) / len(verdicts)
            if s < 0.5:
                print(f"{pe} is not a spyware.")
            else:
                print(f"{pe} is a spyware.")

        except Exception as e:
            print(f"Error processing {pe}: {e}")



# Council accuracy is 97.00%

# Confusion Matrix:
# [[1764   30]
#  [  84 1916]]

# Classification Report:
#               precision    recall  f1-score   support

#            0     0.9545    0.9833    0.9687      1794
#            1     0.9846    0.9580    0.9711      2000

#     accuracy                         0.9700      3794
#    macro avg     0.9696    0.9706    0.9699      3794
# weighted avg     0.9704    0.9700    0.9700      3794
