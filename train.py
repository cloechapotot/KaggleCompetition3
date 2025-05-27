import re, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


STOPWORDS = set(stopwords.words('english'))
LEMMA     = WordNetLemmatizer()

def clean_data(df, text_column='text'):
    df = df.copy()
    def _clean(txt):
        txt = str(txt).lower()
        txt = re.sub(r'http\S+|www\.\S+', '', txt)
        txt = txt.translate(str.maketrans('', '', string.punctuation))
        toks = [t for t in txt.split() if t not in STOPWORDS]
        return ' '.join(LEMMA.lemmatize(t) for t in toks)
    df[text_column] = df[text_column].apply(_clean)
    return df

def remove_outliers(df, text_column='text', min_len=3, max_len=50):
    df = df.copy()
    lengths = df[text_column].str.split().apply(len)
    mask = (lengths >= min_len) & (lengths <= max_len)
    return df[mask].reset_index(drop=True)


# Cell 4 — apply cleaning & outlier removal
train = clean_data(train)
train = remove_outliers(train)
test  = clean_data(test)

# pack texts & labels
X = train['text'].tolist()
y = train['label'].tolist()
X_test = test['text'].tolist()


# Cell 5 — vocab & encoding
def build_vocab(texts, max_vocab_size=10_000):
    ctr = Counter()
    for t in texts: ctr.update(t.split())
    most = ctr.most_common(max_vocab_size-2)
    vocab = {w:i+2 for i,(w,_) in enumerate(most)}
    vocab['<PAD>'], vocab['<OOV>'] = 0, 1
    return vocab

def encode_text(texts, vocab, max_len=50):
    seqs = []
    for t in texts:
        toks = t.split()
        idxs = [vocab.get(w, vocab['<OOV>']) for w in toks]
        if len(idxs)<max_len:
            idxs += [vocab['<PAD>']] * (max_len-len(idxs))
        else:
            idxs = idxs[:max_len]
        seqs.append(idxs)
    return torch.tensor(seqs, dtype=torch.long)

vocab = build_vocab(X)
X_enc = encode_text(X, vocab)
X_test_enc = encode_text(X_test, vocab)


# Cell 6 — train/val split + DataLoaders + label encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_enc, y_enc,
    test_size=0.2, random_state=42, stratify=y_enc
)
y_tr = torch.tensor(y_tr, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]

# we'll rebuild these in Optuna with different batch_sizes
base_loader = lambda X, y, bs, shuffle: DataLoader(
    TextDataset(X,y), batch_size=bs, shuffle=shuffle)

# class-weights
counts = Counter(y_tr.numpy())
weights = torch.tensor([1.0/counts[i] for i in range(num_classes)],
                       dtype=torch.float).to(device)

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 output_dim, pad_idx, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim,
                          bidirectional=True,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


# Cell 8 — Optuna objective + study
def objective(trial):
    # suggest
    embed_dim    = trial.suggest_categorical('embed_dim', [50,100,200])
    hidden_dim   = trial.suggest_categorical('hidden_dim',[64,128,256])
    dropout_rate = trial.suggest_uniform('dropout_rate',0.2,0.6)
    lr           = trial.suggest_loguniform('lr',1e-4,1e-2)
    batch_size   = trial.suggest_categorical('batch_size',[32,64,128])

    # data
    tr_loader = base_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader= base_loader(X_val,y_val,batch_size, shuffle=False)

    # model, opt, crit
    model = MyModel(len(vocab),embed_dim,hidden_dim,
                    num_classes,vocab['<PAD>'],dropout_rate).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss(weight=weights)

    # train 5 epochs
    for _ in range(5):
        model.train()
        for Xb,yb in tr_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()

    # validate
    model.eval()
    preds=[]
    with torch.no_grad():
        for Xb,yb in val_loader:
            Xb = Xb.to(device)
            preds.extend(model(Xb).argmax(dim=1).cpu().numpy())

    return f1_score(y_val.numpy(), np.array(preds), average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(">> Best params:", study.best_params)
print(">> Best val F1:", study.best_value)


# Cell 9 — final training w/ best hyperparams
best = study.best_params

# unpack
embed_dim, hidden_dim = best['embed_dim'], best['hidden_dim']
dropout, lr = best['dropout_rate'], best['lr']
batch_size = best['batch_size']
epochs     = 10

# rebuild loaders
train_loader = base_loader(X_tr, y_tr, batch_size, shuffle=True)
val_loader   = base_loader(X_val, y_val, batch_size, shuffle=False)
test_loader  = DataLoader(TextDataset(X_test_enc), batch_size=batch_size)

# model / optimizer / criterion / scheduler
model = MyModel(len(vocab), embed_dim, hidden_dim,
                num_classes, vocab['<PAD>'], dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(weight=weights)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=1, verbose=True)

best_f1 = 0.0
patience=0
for ep in range(1, epochs+1):
    # train
    model.train()
    total_loss=0
    for Xb,yb in train_loader:
        Xb,yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*Xb.size(0)
    train_loss = total_loss/len(train_loader.dataset)

    # val
    model.eval(); total_loss=0; preds=[]
    with torch.no_grad():
        for Xb,yb in val_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            total_loss += criterion(out,yb).item()*Xb.size(0)
            preds.extend(out.argmax(dim=1).cpu().numpy())
    val_loss = total_loss/len(val_loader.dataset)
    val_f1  = f1_score(y_val.numpy(), np.array(preds), average='macro')
    scheduler.step(val_f1)
    print(f"Epoch {ep}/{epochs} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1, patience = val_f1, 0
        torch.save(model.state_dict(),'best_model.pt')
    else:
        patience+=1
        if patience>=3:
            print("Early stopping.")
            break


# Cell 10 — plots & confusion matrix
# learning curves (optional: collect losses into lists above)
# confusion
model.load_state_dict(torch.load('best_model.pt'))
model.eval(); preds=[]
with torch.no_grad():
    for Xb,yb in val_loader:
        preds.extend(model(Xb.to(device)).argmax(dim=1).cpu().numpy())

cm = confusion_matrix(y_val.numpy(), np.array(preds), labels=range(num_classes))
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.show()


# Cell 11 — inference & submission
test_preds=[]
with torch.no_grad():
    for Xb in test_loader:
        test_preds.extend(model(Xb.to(device)).argmax(dim=1).cpu().numpy())

submission = pd.DataFrame({
    'ID': test.index,   # or test.index if you have an ID column
    'label': le.inverse_transform(test_preds)
})
submission.to_csv('submission_optuna_improved3.csv', index=False)
print("submission_optuna.csv written")
