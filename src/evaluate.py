import torch
import os
import pandas as pd

def evaluate_and_predict(model, test_loader, device, model_dir, output_dir, threshold=0.5):
    weights_path = os.path.join(model_dir, 'best_model.pth')
    test_csv_path = os.path.join(output_dir, 'test.csv')

    # Load best weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    predictions =[]

    print(f"Running inference on test set with tuned threshold of {threshold:.2f}...")
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Apply our newly found optimal threshold
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probs >= threshold).int().cpu().numpy()

            for fname, label in zip(filenames, predicted):
                predictions.append([fname, label])

    # Convert predictions to dataframe
    pred_df = pd.DataFrame(predictions, columns=["filename", "label"])

    # Load the existing test file from outputs/
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Expected to find {test_csv_path} to overwrite, but it does not exist.")
        
    test_df = pd.read_csv(test_csv_path)

    # Fill labels using filename matching
    test_df["label"] = test_df["filename"].map(
        pred_df.set_index("filename")["label"]
    )

    # Overwrite the same file in outputs/
    test_df.to_csv(test_csv_path, index=False)
    print(f"Predictions written into {test_csv_path}")
