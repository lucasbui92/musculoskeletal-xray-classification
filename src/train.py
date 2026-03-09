import torch, os, copy

import numpy as np
from sklearn.metrics import cohen_kappa_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_dir, num_epochs=10):
    best_kappa = -1.0
    best_threshold = 0.5 # Default fallback
    best_model_wts = copy.deepcopy(model.state_dict())
    save_path = os.path.join(model_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        # --- TRAIN PHASE ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # --- VALIDATION PHASE (WITH THRESHOLD TUNING) ---
        model.eval()
        val_loss, val_total = 0.0, 0
        all_val_probs, all_val_labels = [],[]

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                
                # Get raw probabilities for Class 1 (Positive/Abnormal) using Softmax
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / val_total
        
        # --- FIND THE BEST THRESHOLD FOR KAPPA ---
        all_val_probs = np.array(all_val_probs)
        all_val_labels = np.array(all_val_labels)
        
        epoch_best_kappa = -1.0
        epoch_best_thresh = 0.5
        
        # Test every threshold from 0.10 to 0.90 in steps of 0.02
        thresholds = np.arange(0.1, 0.91, 0.02)
        unique_true = np.unique(all_val_labels)

        for thresh in thresholds:
            preds = (all_val_probs >= thresh).astype(int)
            unique_pred = np.unique(preds)

            if len(unique_true) < 2 or len(unique_pred) < 2:
                continue
            kappa = cohen_kappa_score(all_val_labels, preds)
            if np.isnan(kappa):
                continue
            if kappa > epoch_best_kappa:
                epoch_best_kappa = kappa
                epoch_best_thresh = thresh
                
        # Calculate accuracy strictly for logging (using the best found threshold)
        best_preds = (all_val_probs >= epoch_best_thresh).astype(int)
        val_acc = 100 * (best_preds == all_val_labels).mean()

        print(f"Epoch[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Kappa: {epoch_best_kappa:.4f} (Best Thresh: {epoch_best_thresh:.2f})")

        # Save best model based on the Tuned Kappa score
        if epoch_best_kappa > best_kappa:
            best_kappa = epoch_best_kappa
            best_threshold = epoch_best_thresh
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best Validation Kappa: {best_kappa:.4f} achieved at Threshold {best_threshold:.2f}")
    
    # Return both the weights AND the best threshold
    return best_model_wts, best_threshold
