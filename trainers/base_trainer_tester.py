"""
Base training functionality for all methods with multi-label support.
"""
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import hamming_loss, jaccard_score
import torch.nn.functional as F

def train(args, trial, models, criterion, optimizers, schedulers, dataloaders, writer=None):
    """
    Standard training loop for models that don't require special handling.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: loss function
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object (optional)
        
    Returns:
        None
    """
    if writer is None:
        log_dir = f'logs/tensorboard/{args.dataset}/{args.method}/{trial}_experiment'
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
        if args.textset:  # text dataset
            if not args.causal_lm:
                epoch_loss, epoch_accuracy = train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch)
            else:
                epoch_loss, epoch_accuracy = train_epoch_nlp_casuallm(args, models, criterion, optimizers, dataloaders, writer, epoch)

        else:
            epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
        
        schedulers['backbone'].step()
        writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
        writer.add_scalar('training_loss', epoch_loss, epoch)
        writer.add_scalar('accuracy', epoch_accuracy, epoch)

    writer.close()


def train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch):
    """
    Standard training epoch for regular supervised learning.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        criterion: loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object
        epoch: current epoch number
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    models['backbone'].train()
    is_multilabel = getattr(args, 'is_multilabel', False)

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_batches = len(dataloaders['train'])

    for i, data in enumerate(dataloaders['train']):
        inputs, labels = data[0].to(args.device), data[1].to(args.device)

        optimizers['backbone'].zero_grad()

        scores, _ = models['backbone'](inputs)
        
        if is_multilabel:
            # For multi-label, labels are already float tensors
            labels = labels.float()
            target_loss = criterion(scores, labels)
            m_backbone_loss = torch.mean(target_loss)
        else:
            # For single-label classification
            target_loss = criterion(scores, labels)
            # Use torch.mean to handle both scalar and tensor cases
            m_backbone_loss = torch.mean(target_loss)
        
        loss = m_backbone_loss
        loss.backward()
        optimizers['backbone'].step()

        running_loss += loss.item()

        # Calculate accuracy based on task type
        if is_multilabel:
            # Multi-label accuracy: exact match ratio
            preds = (torch.sigmoid(scores) > 0.5).float()
            correct_predictions += torch.sum(torch.all(preds == labels, dim=1)).item()
        else:
            # Single-label accuracy
            _, preds = torch.max(scores, 1)
            correct_predictions += torch.sum(preds == labels).item()
        
        total_predictions += labels.size(0)

        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            writer.add_scalar('training_loss_batch', avg_loss, epoch * total_batches + i)
            running_loss = 0.0

    epoch_loss = running_loss / total_batches
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch):
    """
    Training epoch for NLP models (BERT/RoBERTa variants) with multi-label support.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        criterion: loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object
        epoch: current epoch number
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    models['backbone'] = models['backbone'].to(args.device)
    models['backbone'].train()
    is_multilabel = getattr(args, 'is_multilabel', False)

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_batches = len(dataloaders['train'])

    for i, data in enumerate(dataloaders['train']):
        # Extract input_ids, attention_mask, and labels from the dictionary
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        labels = data['labels'].to(args.device)

        # Zero the gradients
        optimizers['backbone'].zero_grad()

        # Forward pass
        outputs = models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
        scores = outputs.logits  # For BertForSequenceClassification, logits contain class probabilities

        # Compute loss based on task type
        if is_multilabel:
            # For multi-label, ensure labels are float
            labels = labels.float()
            target_loss = criterion(scores, labels)
            m_backbone_loss = torch.mean(target_loss)
        else:
            # For single-label classification
            target_loss = criterion(scores, labels)
            # Use torch.mean to handle both scalar and tensor cases
            m_backbone_loss = torch.mean(target_loss)

        loss = m_backbone_loss

        # Backward pass and optimization
        loss.backward()
        optimizers['backbone'].step()

        # Update running loss
        running_loss += loss.item()

        # Calculate predictions and accuracy based on task type
        if is_multilabel:
            # Multi-label accuracy: exact match ratio
            preds = (torch.sigmoid(scores) > 0.5).float()
            correct_predictions += torch.sum(torch.all(preds == labels, dim=1)).item()
        else:
            # Single-label accuracy
            _, preds = torch.max(scores, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
        
        total_predictions += labels.size(0)

        # Log training loss every 100 batches
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            writer.add_scalar('training_loss_batch', avg_loss, epoch * total_batches + i)
            running_loss = 0.0

    # Calculate epoch metrics
    epoch_loss = running_loss / total_batches
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy

def train_epoch_nlp_casuallm(args, models, criterion, optimizers, dataloaders, writer, epoch):
    model = models["backbone"]
    tokenizer = models.get("tokenizer", None)
    model.train()

    device = args.device
    train_loader = dataloaders["train_in"]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    option_texts = ["(A)", "(B)", "(C)", "(D)"]
    tok = tokenizer(option_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    option_tokens = [int(t[0]) for t in tok["input_ids"]]

    option_tokens = torch.tensor(option_tokens, device=device)  # [4]

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)           # [B, T]
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)                 # [B, T], prompt = -100
        option_id = batch["option_id"].to(device)           # [B], 0~3

        optimizers["backbone"].zero_grad()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        optimizers["backbone"].step()

        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

        with torch.no_grad():
            # out.logits: [B, T, vocab]
            last_logits = out.logits[:, -1, :]               # [B, vocab]
            abcd_logits = last_logits.index_select(1, option_tokens)  # [B, 4]
            pred = abcd_logits.argmax(dim=1)                 # [B]
            total_correct += (pred == option_id).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    print(f"[Train causal LM] Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    return avg_loss, avg_acc


def test(args, models, dataloaders):
    """
    Test the model on the test set with enhanced metrics supporting multi-label.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Dictionary containing accuracy, precision, recall and f1 score
    """
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Switch to evaluate mode
    models['backbone'].eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # Compute output
            with torch.no_grad():
                if args.method == 'TIDAL':
                    scores, _, _ = models['backbone'](inputs, method='TIDAL')
                else:
                    scores, _ = models['backbone'](inputs)

            # Get predictions based on task type
            if is_multilabel:
                preds = (torch.sigmoid(scores) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            else:
                _, preds = torch.max(scores.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Convert to appropriate format
    if is_multilabel:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate multi-label metrics
        exact_match = accuracy_score(all_labels, all_preds)
        hamming = hamming_loss(all_labels, all_preds)
        jaccard = jaccard_score(all_labels, all_preds, average='samples')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print('Multi-label Test Results:')
        print(f'* Exact Match Ratio: {exact_match:.3f}')
        print(f'* Hamming Loss: {hamming:.3f}')
        print(f'* Jaccard Score: {jaccard:.3f}')
        print(f'* Macro Precision: {precision:.3f}')
        print(f'* Macro Recall: {recall:.3f}')
        print(f'* Macro F1 Score: {f1:.3f}')
        
        return exact_match, precision, recall, f1
    else:
        # Convert lists to numpy arrays for single-label
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate single-label metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print('Single-label Test Results:')
        print(f'* Accuracy: {accuracy:.3f}')
        print(f'* Precision: {precision:.3f}')
        print(f'* Recall: {recall:.3f}')
        print(f'* F1 Score: {f1:.3f}')
        
        return accuracy, precision, recall, f1


def test_nlp(args, models, dataloaders):
    """
    Test NLP models on the test set with enhanced metrics supporting multi-label.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Dictionary containing accuracy, precision, recall and f1 score
    """
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Switch to evaluation mode
    models['backbone'].eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            # Extract and move data to the correct device
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)
            
            scores = models['backbone'](
                input_ids=input_ids, 
                attention_mask=attention_mask
            ).logits  # Get logits from BertForSequenceClassification
            
            # Get predictions based on task type
            if is_multilabel:
                preds = (torch.sigmoid(scores) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            else:
                _, preds = torch.max(scores.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics based on task type
    if is_multilabel:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate multi-label metrics
        exact_match = accuracy_score(all_labels, all_preds)
        hamming = hamming_loss(all_labels, all_preds)
        jaccard = jaccard_score(all_labels, all_preds, average='samples')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print('Multi-label NLP Test Results:')
        print(f'* Exact Match Ratio: {exact_match:.3f}')
        print(f'* Hamming Loss: {hamming:.3f}')
        print(f'* Jaccard Score: {jaccard:.3f}')
        print(f'* Macro Precision: {precision:.3f}')
        print(f'* Macro Recall: {recall:.3f}')
        print(f'* Macro F1 Score: {f1:.3f}')
        
        return exact_match, precision, recall, f1
    else:
        # Convert lists to numpy arrays for single-label
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate single-label metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print('Single-label NLP Test Results:')
        print(f'* Accuracy: {accuracy:.3f}')
        print(f'* Precision: {precision:.3f}')
        print(f'* Recall: {recall:.3f}')
        print(f'* F1 Score: {f1:.3f}')
        
        return accuracy, precision, recall, f1

def test_nlp_casuallm(args, models, dataloaders):
    """
    Test for causal LM (e.g., Llama) using single-character options A/B/C/D.
    We read the last-token logits, extract A/B/C/D columns, and compute metrics.
    
    Returns:
        (accuracy, precision, recall, f1) with single-label 'weighted' averages.
    """
    device = args.device
    model = models["backbone"].to(device)
    tokenizer = models.get("tokenizer", None)
    assert tokenizer is not None, "tokenizer is required for causal LM testing."

    model.eval()

    cand_sets = [
        ["A", "B", "C", "D"],
        ["▁A", "▁B", "▁C", "▁D"],
    ]
    opt_ids = None
    for cand in cand_sets:
        ids = tokenizer.convert_tokens_to_ids(cand)
        if all(i is not None and i != tokenizer.unk_token_id for i in ids):
            opt_ids = torch.tensor(ids, device=device)
            break
    if opt_ids is None:
        cand_enc = tokenizer(["A", "B", "C", "D"], add_special_tokens=False, return_tensors="pt", padding=True)
        ids = [int(row[0].item()) for row in cand_enc["input_ids"]]
        opt_ids = torch.tensor(ids, device=device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloaders["test"]:
            input_ids = batch["input_ids"].to(device)            # [B, T]
            attention_mask = batch["attention_mask"].to(device)  # [B, T]

            # 2) 前向，取最後一個 token 的 logits
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits: [B, T, vocab] → 取最後位置
            last_logits = out.logits[:, -1, :]                    # [B, vocab]

            # 3) 只抽 A/B/C/D 的 logits，softmax 得到分布
            abcd_logits = last_logits.index_select(dim=1, index=opt_ids)  # [B, 4]
            probs = F.softmax(abcd_logits, dim=1)                         # [B, 4]
            preds = probs.argmax(dim=1)                                    # [B] in {0,1,2,3}

            # 4) 取得真實標籤：
            #    主要來源：dataset 提供的 'option_id'（建議在 causal LM dataset 中回傳）
            if "option_id" in batch:
                labels = batch["option_id"].to(device)                     # [B]
            else:
                # 後備方案：從 labels（-100 mask）中找最後一個非 -100 的 token，對應到 A/B/C/D
                # 若對不上就跳過該筆
                labels = []
                if "labels" in batch:
                    lab = batch["labels"].to(device)                       # [B, T]
                    for i in range(lab.size(0)):
                        row = lab[i]
                        idxs = (row != -100).nonzero(as_tuple=False).squeeze(-1)
                        if idxs.numel() == 0:
                            labels.append(-1)
                            continue
                        last_idx = idxs[-1].item()
                        tok_id = int(row[last_idx].item())
                        # map tok_id → {0,1,2,3}
                        if tok_id in opt_ids.tolist():
                            labels.append(opt_ids.tolist().index(tok_id))
                        else:
                            labels.append(-1)
                    labels = torch.tensor(labels, device=device)
                else:
                    labels = torch.full((preds.size(0),), -1, device=device, dtype=torch.long)

            valid_mask = labels != -1
            if valid_mask.any():
                all_preds.extend(preds[valid_mask].detach().cpu().numpy().tolist())
                all_labels.extend(labels[valid_mask].detach().cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if all_labels.size == 0:
        print("Warning: no valid labels found in test_nlp_casuallm; returning zeros.")
        return 0.0, 0.0, 0.0, 0.0

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("Causal-LM NLP Test Results (A/B/C/D):")
    print(f"* Accuracy:  {accuracy:.3f}")
    print(f"* Precision: {precision:.3f}")
    print(f"* Recall:    {recall:.3f}")
    print(f"* F1 Score:  {f1:.3f}")

    return accuracy, precision, recall, f1

def test_ood(args, models, dataloaders):
    """
    Test OOD detection models on the test set with enhanced metrics.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'ood_detection' is the OOD model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Dictionary containing accuracy, precision, recall and f1 score
    """
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Switch to evaluate mode
    models['ood_detection'].eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            
            # Compute output
            scores, _ = models['ood_detection'](inputs)
            
            # Get predictions
            _, preds = torch.max(scores.data, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print('OOD Detection Test Results:')
    print(f'* Accuracy: {accuracy:.3f}')
    print(f'* Precision: {precision:.3f}')
    print(f'* Recall: {recall:.3f}')
    print(f'* F1 Score: {f1:.3f}')
    
    return accuracy, precision, recall, f1


def test_ood_nlp(args, models, dataloaders):
    """
    Test OOD detection NLP models on the test set with enhanced metrics.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'ood_detection' is the OOD model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Dictionary containing accuracy, precision, recall and f1 score
    """
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Switch to evaluate mode
    models['ood_detection'].eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            # Extract input_ids, attention_mask, and labels from the dictionary
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)
            
            # Compute output
            outputs = models['ood_detection'](
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            scores = outputs.logits
            
            # Get predictions
            _, preds = torch.max(scores.data, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print('OOD Detection Test Results:')
    print(f'* Accuracy: {accuracy:.3f}')
    print(f'* Precision: {precision:.3f}')
    print(f'* Recall: {recall:.3f}')
    print(f'* F1 Score: {f1:.3f}')
    
    return accuracy, precision, recall, f1