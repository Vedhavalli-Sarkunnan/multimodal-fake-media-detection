import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class GatedFusionClassifier(nn.Module):
    def __init__(self, title_dim, body_dim, hidden_dim, num_classes, mode="fusion"):
        super().__init__()

        self.mode = mode
        self.title_encoder = nn.Sequential(
            nn.Linear(title_dim, hidden_dim),
            nn.ReLU(), # ReLU = max(0, x)
            nn.Dropout(0.3)
        )
        
        self.body_encoder = nn.Sequential(
            nn.Linear(body_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        if mode == "fusion":
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, title_emb=None, body_emb=None):
        
        if self.mode == "title_only":
            h = self.title_encoder(title_emb)

        elif self.mode == "body_only":
            h = self.body_encoder(body_emb)

        else: #fusion
            h_title = self.title_encoder(title_emb)
            h_body = self.body_encoder(body_emb)

            gate_input = torch.cat([h_title, h_body], dim=1)
            alpha = self.gate(gate_input)

            h = alpha * h_title + h_body

        logits = self.classifier(h)
        return logits

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none"
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)     # per-sample loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


from copy import deepcopy

def train_gated_fusion_model(
    x_title_train,
    x_body_train,
    y_train,
    x_title_val,
    x_body_val,
    y_val,
    TITLE_DIM,
    BODY_DIM,
    mode="fusion",
    hidden_dim=256,
    epochs=50,
    batch_size=64,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    if mode == "fusion":
        x_title_train_t = torch.tensor(x_title_train, dtype=torch.float32)
        x_body_train_t = torch.tensor(x_body_train, dtype=torch.float32)

        x_title_val_t = torch.tensor(x_title_val, dtype=torch.float32)
        x_body_val_t = torch.tensor(x_body_val, dtype=torch.float32)

        train_ds = TensorDataset(x_title_train_t, x_body_train_t, y_train_t)
        val_ds = TensorDataset(x_title_val_t, x_body_val_t, y_val_t)

    elif mode == "title_only":
        x_title_train_t = torch.tensor(x_title_train, dtype=torch.float32)
        x_title_val_t   = torch.tensor(x_title_val, dtype=torch.float32)

        train_ds = TensorDataset(x_title_train_t, y_train_t)
        val_ds   = TensorDataset(x_title_val_t,   y_val_t)

    else:  # body_only
        x_body_train_t = torch.tensor(x_body_train, dtype=torch.float32)
        x_body_val_t   = torch.tensor(x_body_val, dtype=torch.float32)

        train_ds = TensorDataset(x_body_train_t, y_train_t)
        val_ds   = TensorDataset(x_body_val_t,   y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = GatedFusionClassifier(
        title_dim=TITLE_DIM,
        body_dim=BODY_DIM,
        hidden_dim=hidden_dim,
        num_classes=2,
        mode=mode
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = WeightedFocalLoss(class_weights=class_weights, gamma=2)
  
    best_f1 = 0
    best_model = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            if mode == "fusion":
                x_title, x_body, y = batch
                logits = model(x_title.to(device), x_body.to(device))
    
            elif mode == "title_only":
                x_title, y = batch
                logits = model(title_emb=x_title.to(device))
    
            else: #body_only
                x_body, y = batch
                logits = model(body_emb=x_body.to(device))
    
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                if mode == "fusion":
                    x_title, x_body, y = batch
                    logits = model(x_title.to(device), x_body.to(device))
    
                elif mode == "title_only":
                    x_title, y = batch
                    logits = model(title_emb=x_title.to(device))
    
                else: 
                    x_body, y = batch
                    logits = model(body_emb=x_body.to(device))
    
                probs = torch.softmax(logits, dim=1)[:, 1]   # P(fake)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} | Val F1: {val_f1:.4f}")

        #Early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
                    
    print(f"[{mode.upper()}] Best Validation F1 Score: {best_f1:.4f}")
    
    return best_model, best_f1
