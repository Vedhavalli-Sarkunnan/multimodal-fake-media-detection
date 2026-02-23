from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_logistic_regression(x_train, y_train, x_val, y_val, C=0.5):

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=C, # regularization strength to prevent overfitting
        penalty="l2", #L2 regularization
        solver="liblinear" #optimizer
    )
    model.fit(x_train, y_train)
    val_preds = model.predict(x_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation accuracy for Logistic Regression: {val_acc}")
    print(f"Validation F1 score for Logistic Regression: {val_f1}")
    
    return model, val_f1

def train_xgboost(x_train, y_train, x_val, y_val, max_depth=3, n_estimators=50, learning_rate=0.1, early_stopping_rounds=20):

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        objective="binary:logistic", #sigmoid function to predict probability between 0 and 1
        eval_metric="logloss", #Binary Cross Entropy (loss function)
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8, #each tree uses 80% of the training data 
        colsample_bytree=0.8, #each tree uses 80% of the features
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1
    )
    
    model.fit(
      x_train, y_train,
      eval_set=[(x_val, y_val)],
      verbose=10
    )
    
    val_preds = model.predict(x_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation accuracy for XGBoost: {val_acc}")
    print(f"Validation F1 score for XGBoost: {val_f1}")
    
    return model, val_f1

def train_LinearSVC(x_train, y_train, x_val, y_val, C=0.7, random_state=16):

    model = LinearSVC(
        C=C, #regularization strength to prevent/reduce overfitting
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(x_train, y_train)
    val_preds = model.predict(x_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation accuracy for LinearSVC: {val_acc}")
    print(f"Validation F1 score for LinearSVC: {val_f1}")
    
    return model, val_f1