from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def create_validation_split(data, val_size=0.1, seed=42):
    train_idx, val_idx = train_test_split(
        range(len(data)),
        test_size=val_size,
        stratify=data['label'],
        random_state=seed
    )
    return train_idx, val_idx