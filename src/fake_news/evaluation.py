import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import numpy as np

def store_results(model_name, experiment_type, metrics_dict, json_path="results/fake_news_results.json"):
    # Load existing results
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Ensure model key exists
    if model_name not in results:
        results[model_name] = {}

    # Store experiment
    results[model_name][experiment_type] = metrics_dict

    # Write back
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Stored results → Model: {model_name}, Experiment: {experiment_type}")

def get_model_probs(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))  # sigmoid

    else:
        raise ValueError("Model does not support probability output")

#Used only for classical ML models 
def evaluate_model(
    model,
    x_test,
    y_test,
    model_name,
    experiment_name,
    threshold,
    json_path="results/fake_news_results.json",
    is_test=True,
    return_probs=False
):
    probs = get_model_probs(model, x_test)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)
    cr  = classification_report(y_test, preds, output_dict=True)

    if is_test:
        experiment_results = {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": cr
        }

        store_results(
            model_name=model_name,
            experiment_type=experiment_name,
            metrics_dict=experiment_results,
            json_path=json_path
        )

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_test, preds))

    if return_probs:
        return probs, np.array(y_test)

def tune_threshold_and_eval_classical(
    model,
    x_val,
    y_val,
    x_test,
    y_test,
    model_name,
    experiment_name,
    threshold_dict,
    json_path="results/fake_news_results.json"
):
    # Initialize nested dict if needed
    if model_name not in threshold_dict:
        threshold_dict[model_name] = {}

    # Use stored threshold if exists
    if experiment_name in threshold_dict[model_name]:
        threshold = threshold_dict[model_name][experiment_name]
        print(f"Using stored threshold: {threshold:.2f}")

    else:
        val_probs, val_labels = evaluate_model(
            model=model,
            x_test=x_val,
            y_test=y_val,
            model_name=model_name,
            experiment_name=experiment_name,
            threshold=0.5,
            is_test=False,
            return_probs=True
        )

        threshold, best_f1 = find_best_threshold(val_probs, val_labels)
        threshold_dict[model_name][experiment_name] = threshold

        print(f"Best threshold found: {threshold:.2f} | Val Macro F1: {best_f1:.4f}")

    # Final evaluation on TEST
    evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        model_name=model_name,
        experiment_name=experiment_name,
        threshold=threshold,
        json_path=json_path,
        is_test=True
    )
        
def normalize_weights(weights):
    for model in weights:
        total = sum(weights[model].values()) # sum of F1 scores
        for dataset_type in weights[model]:
            weights[model][dataset_type]/= total
    return weights

def get_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[0, 1]
    elif hasattr(model, "decision_function"):
        return 1 / (1 + np.exp(-model.decision_function(x)[0]))
    else:
        return float(model.predict(x)[0])

import torch
import numpy as np

def get_dl_score(model, x_title=None, x_body=None, mode="fusion"):

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if mode == "fusion":
            logits = model(
                torch.tensor(x_title, dtype=torch.float32).to(device),
                torch.tensor(x_body, dtype=torch.float32).to(device)
            )
        elif mode == "title_only":
            logits = model(
                title_emb=torch.tensor(x_title, dtype=torch.float32).to(device)
            )

        else:  # body_only
            logits = model(
                body_emb=torch.tensor(x_body, dtype=torch.float32).to(device)
            )
        probs = torch.softmax(logits, dim=1)
        return probs[:,1].item()
    
def evaluate_mixed_test_dataset_dl(x_test, y_test, availability_mask, models, weights, model_name, TITLE_DIM, BODY_DIM, json_path, threshold=0.5):
    preds = []
    weights = normalize_weights(weights)

    for i in range(len(x_test)):
        x = x_test[i].reshape(1, -1)
        has_title, has_body = availability_mask[i]
        embedding_dim = x.shape[1]

        votes = []
        vote_weights = []

        if has_title and has_body and embedding_dim == TITLE_DIM + BODY_DIM:
            score = get_dl_score(
                model=models[model_name]["full"],
                x_title=x[:, :TITLE_DIM],
                x_body=x[:, TITLE_DIM:],
                mode="fusion"
            )
            votes.append(score)
            vote_weights.append(weights[model_name]["full"])

        if has_title:
            if embedding_dim == TITLE_DIM:
                x_title = x
            elif embedding_dim == TITLE_DIM + BODY_DIM:
                x_title = x[:, :TITLE_DIM]
            else:
                x_title = None

            if x_title is not None:
                score = get_dl_score(
                    model=models[model_name]["title"],
                    x_title=x_title,
                    mode="title_only"
                )
                votes.append(score)
                vote_weights.append(weights[model_name]["title"])
                
        if has_body:
            if embedding_dim == BODY_DIM:
                x_body = x
            elif embedding_dim == TITLE_DIM + BODY_DIM:
                x_body = x[:, TITLE_DIM:]
            else:
                x_body = None

            if x_body is not None:
                score = get_dl_score(
                    model=models[model_name]["body"],
                    x_body=x_body,
                    mode="body_only"
                )
                votes.append(score)
                vote_weights.append(weights[model_name]["body"])

        final_score = np.average(votes, weights=vote_weights)
        final_pred = int(final_score >= threshold)
        preds.append(final_pred)

    preds = np.array(preds)
    test_acc = accuracy_score(y_test, preds)
    test_f1  = f1_score(y_test, preds)
    test_cm  = confusion_matrix(y_test, preds)
    test_cr  = classification_report(y_test, preds, output_dict=True)

    print(f"Accuracy for {model_name} (DL ensemble): {test_acc}")
    print(f"F1 score for {model_name} (DL ensemble): {test_f1}")
    print(f"Confusion Matrix:\n{test_cm}")
    print(f"Classification Report:\n{classification_report(y_test, preds)}")

    experiment_results = {
    "accuracy": test_acc,
    "f1_score": test_f1,
    "confusion_matrix": test_cm.tolist(),
    "classification_report": test_cr
    }
    
    store_results(
        model_name=model_name,
        experiment_type="DL_weighted_ensemble",
        metrics_dict=experiment_results,
        json_path=json_path
    )

def evaluate_mixed_test_dataset(
    x_test,
    y_test,
    availability_mask,
    models,
    weights,
    model_name,
    TITLE_DIM,
    BODY_DIM,
    json_path
):
    preds = []
    weights = normalize_weights(weights)

    for i in range(len(x_test)):
        x = x_test[i].reshape(1, -1)
        has_title, has_body = availability_mask[i]

        votes = []
        vote_weights = []

        embedding_dim = x.shape[1]

        # Dataset has both title and body
        if has_title and has_body and embedding_dim == TITLE_DIM + BODY_DIM:
            score = get_score(models[model_name]["full"], x)
            votes.append(score)
            vote_weights.append(weights[model_name]["full"])

        # Dataset has title
        if has_title:
            x_title = (
                x if embedding_dim == TITLE_DIM
                else x[:, :TITLE_DIM] if embedding_dim == TITLE_DIM + BODY_DIM
                else None
            )

            if x_title is not None:
                score = get_score(models[model_name]["title"], x_title)
                votes.append(score)
                vote_weights.append(weights[model_name]["title"])

        # Dataset has body
        if has_body:
            x_body = (
                x if embedding_dim == BODY_DIM
                else x[:, TITLE_DIM:] if embedding_dim == TITLE_DIM + BODY_DIM
                else None
            )

            if x_body is not None:
                score = get_score(models[model_name]["body"], x_body)
                votes.append(score)
                vote_weights.append(weights[model_name]["body"])

        final_score = np.average(votes, weights=vote_weights)
        final_pred = int(final_score >= 0.5)
        preds.append(final_pred)

    preds = np.array(preds)

    test_acc = accuracy_score(y_test, preds)
    test_f1  = f1_score(y_test, preds)
    test_cm  = confusion_matrix(y_test, preds)
    test_cr  = classification_report(y_test, preds, output_dict=True)

    print(f"Accuracy for {model_name}: {test_acc}")
    print(f"Confusion Matrix:\n{test_cm}")
    print(classification_report(y_test, preds))

    store_results(
        model_name=model_name,
        experiment_type="ML_weighted_ensemble",
        metrics_dict={
            "accuracy": test_acc,
            "f1_score": test_f1,
            "confusion_matrix": test_cm.tolist(),
            "classification_report": test_cr
        },
        json_path=json_path
    )

import torch
from torch.utils.data import DataLoader, TensorDataset

def find_best_threshold(probs, labels):
    
    thresholds = np.linspace(0.1, 0.9, 81) #step value of 0.01 => [0.10, 0.11, 0.12, ...., 0.90] => 81 values
    
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print("Using macro f1 score for threshold")
    return best_threshold, best_f1

def evaluate_dl_model(
    model,
    x_title_test,
    x_body_test,
    y_test,
    model_name,
    mode="fusion",
    batch_size=64,
    json_path="results/fake_news_results.json",
    threshold=None,
    return_probs=False,
    is_test=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_test_t = torch.tensor(y_test, dtype=torch.long)

    if mode == "fusion":
        x_title_t = torch.tensor(x_title_test, dtype=torch.float32)
        x_body_t = torch.tensor(x_body_test, dtype=torch.float32)
        dataset = TensorDataset(x_title_t, x_body_t, y_test_t)

    elif mode == "title_only":
        x_title_t = torch.tensor(x_title_test, dtype=torch.float32)
        dataset = TensorDataset(x_title_t, y_test_t)

    else: #body_only
        x_body_t = torch.tensor(x_body_test, dtype=torch.float32)
        dataset = TensorDataset(x_body_t, y_test_t)

    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    if return_probs:
        all_probs = []

    with torch.no_grad():
        for batch in loader:

            if mode == "fusion":
                x_title, x_body, y = batch
                logits = model(x_title.to(device),x_body.to(device))

            elif mode == "title_only":
                x_title, y = batch
                logits = model(title_emb=x_title.to(device))

            else: #body_only
                x_body, y = batch
                logits = model(body_emb=x_body.to(device))

            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            
            if threshold is None:
                preds = np.argmax(logits.cpu().numpy(), axis=1)
            else:
                preds = (probs >= threshold).astype(int)

            if return_probs:
                all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1  = f1_score(all_labels, all_preds)
        test_cm  = confusion_matrix(all_labels, all_preds)
        test_cr  = classification_report(all_labels, all_preds, output_dict=True)

        if is_test:
            experiment_results = {
                "accuracy": test_acc,
                "f1_score": test_f1,
                "confusion_matrix": test_cm.tolist(),
                "classification_report": test_cr
            }
            
            store_results(
                model_name=model_name,
                experiment_type=mode,
                metrics_dict=experiment_results,
                json_path=json_path
            )
    
            print(f"Test accuracy: {test_acc}")
            print(f"Test F1 score: {test_f1}")
            print(f"Confusion Matrix:\n{test_cm}")
            print(f"Classification Report:\n{classification_report(all_labels, all_preds)}")
        if return_probs:
            return np.array(all_probs), np.array(all_labels)

def tune_threshold_and_eval(
    model,
    x_title_val,
    x_body_val,
    y_val,
    x_title_test,
    x_body_test,
    y_test,
    mode,
    threshold_dict,
    model_name,
    json_path
):
    if threshold_dict[model_name][mode] is not None:
        threshold = threshold_dict[model_name][mode]
    else:
        val_probs, val_labels = evaluate_dl_model(
            model=model,
            x_title_test=x_title_val,
            x_body_test=x_body_val,
            y_test=y_val,
            model_name=model_name,
            mode=mode,
            return_probs=True,
            is_test=False
        )
    
        threshold_dict[model_name][mode], _ = find_best_threshold(val_probs, val_labels)

    evaluate_dl_model(
        model=model,
        x_title_test=x_title_test,
        x_body_test=x_body_test,
        y_test=y_test,
        model_name=model_name,
        mode=mode,
        threshold=threshold_dict[model_name][mode],
        is_test=True,
        json_path=json_path
    )
