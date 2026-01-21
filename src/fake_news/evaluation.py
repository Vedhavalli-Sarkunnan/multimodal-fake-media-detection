import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

def store_results(result_dict, json_path="results.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    existing_results.update(result_dict)

    with open(json_path, "w") as f:
        json.dump(existing_results, f, indent=4)

    print(f"Results stored in {json_path}")

def evaluate_model(model, x_test, y_test, experiment_name, json_path="results.json"):

    test_preds = model.predict(x_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    test_cr = classification_report(y_test, test_preds, output_dict=True)

    experiment_results = {
        experiment_name: {
            "accuracy": test_acc,
            "f1_score": test_f1,
            "confusion_matrix": test_cm.tolist(),
            "classification_report": test_cr
        }
    }
    store_results(experiment_results, json_path)
    print(f"Test accuracy: {test_acc}")
    print(f"Test F1 score: {test_f1}")
    print(f"Confusion Matrix:\n {test_cm}")
    print(f"Classification Report:\n {test_cr}")

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

def evaluate_mixed_test_dataset(x_test, y_test, availability_mask, models, weights, model_name, TITLE_DIM, BODY_DIM):
    preds = []
    weights = normalize_weights(weights)
    for i in range(len(x_test)):
        x = x_test[i].reshape(1, -1)
        has_title, has_body = availability_mask[i]
        
        votes = []
        vote_weights = []

        embedding_dim = x.shape[1]
        # FULL MODEL -> only if both title and body exist
        if has_title and has_body and embedding_dim == TITLE_DIM+BODY_DIM:
            score = get_score(models[model_name]["full"], x)
            votes.append(score)
            vote_weights.append(weights[model_name]["full"])

        #TITLE MODEL -> if title exists
        if has_title:
            if embedding_dim == TITLE_DIM:
                x_title = x
            elif embedding_dim == TITLE_DIM+BODY_DIM:
                x_title = x[:, :TITLE_DIM]
            else:
                x_title = None
                
            if x_title is not None:
                title_score = get_score(models[model_name]["title"], x_title)
                votes.append(title_score)
                vote_weights.append(weights[model_name]["title"])

        #BODY MODEL -> if body exist
        if has_body:
            if embedding_dim == BODY_DIM:
                x_body = x
            elif embedding_dim == TITLE_DIM+BODY_DIM:
                x_body = x[:, TITLE_DIM:]
            else:
                x_body = None
                
            if x_body is not None:
                body_score = get_score(models[model_name]["body"], x_body)
                votes.append(body_score)
                vote_weights.append(weights[model_name]["body"])

        final_score = np.average(votes, weights=vote_weights)
        final_pred = int(final_score >= 0.5)
        preds.append(final_pred)

    preds = np.array(preds)
    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds)
    test_cm = confusion_matrix(y_test, preds)
    test_cr = classification_report(y_test, preds,output_dict=True)
    print(f"Accuracy for {model_name}: {test_acc}")
    print(f"Confusion Matrix for {model_name}:\n{test_cm}")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, preds)}")

    experiment_name = model_name + "_mixed_embeddings_weighted_ensemble"
    results = {
        experiment_name: {
          "accuracy": test_acc,
          "f1_score": test_f1,
          "confusion_matrix": test_cm.tolist(),
          "classification_report": test_cr
        }
    }
    store_results(results, "results.json")
