import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleMedicalCNN

def predict_random_scan():
    # 1. Load the final saved model weights
    try:
        model = SimpleMedicalCNN()
        model.load_state_dict(torch.load("federated_model.pth"))
        model.eval()
    except FileNotFoundError:
        print("[ERROR] Could not find 'federated_model.pth'. Run 'run_all.py' with the updated server script first!")
        return

    # 2. Grab a random unseen test scan from your raw NPZ file
    raw_data = np.load('pneumoniamnist.npz')
    X_test = raw_data['test_images']
    y_test = raw_data['test_labels'].squeeze()
    
    random_idx = np.random.randint(0, len(X_test))
    raw_image = X_test[random_idx]
    true_label = y_test[random_idx]

    # 3. Process the image for PyTorch tensor input [Batch=1, Channel=1, H=28, W=28]
    image_tensor = torch.tensor(raw_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    # 4. Run Model Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    # 5. Translate text outputs
    classes = {0: "Healthy (Normal)", 1: "Pneumonia Detected"}
    
    print("\n" + "="*40)
    print(f"🏥 CLINICAL AI DIAGNOSIS REPORT")
    print("="*40)
    print(f"True Patient Condition : {classes[true_label]}")
    print(f"AI Predicted Diagnosis : {classes[prediction]}")
    print("="*40)
    
    if prediction == true_label:
        print("✅ Result: Correct Diagnosis!")
    else:
        print("❌ Result: Incorrect Diagnosis.")

    # 6. Pop up a quick image visualizer so you can see the lung scan
    plt.imshow(raw_image, cmap='gray')
    plt.title(f"AI Prediction: {classes[prediction]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_random_scan()