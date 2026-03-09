import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Import custom modules from the src/ folder
from src.preprocessing import get_transforms
from src.dataloader import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_and_predict

def main():
    data_path = "./sample_dataset/"
    TRAIN_DIR = os.path.join(data_path, "train")
    TEST_DIR = os.path.join(data_path, "test")

    assert os.path.exists(TRAIN_DIR), "Train folder not found"
    assert os.path.exists(TEST_DIR), "Test folder not found"

    OUTPUT_DIR = "./outputs/"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure outputs dir exists
    os.makedirs(MODEL_DIR, exist_ok=True) # Ensure models dir exists

    # Hardware Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get Transforms and DataLoaders
    transform_train, transform_test = get_transforms()
    
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=TRAIN_DIR, 
        test_dir=TEST_DIR, 
        transform_train=transform_train, 
        transform_test=transform_test,
        batch_size=32
    )

    # Initialize Model, Loss, and Optimizer
    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- TRAINING MODEL ---
    print("\nStarting training...")
    best_weights, best_thresh = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=MODEL_DIR, # Models are stored here
        num_epochs=10
    )

    # --- EVALUATE AND PREDICT ---
    evaluate_and_predict(
        model=model,
        test_loader=test_loader,
        device=device,
        model_dir=MODEL_DIR, # Loads best_model.pth from here
        output_dir=OUTPUT_DIR, # Uses outputs/test.csv for predictions
        threshold=best_thresh
    )

if __name__ == '__main__':
    main()
