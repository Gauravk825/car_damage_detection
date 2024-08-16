from ultralytics import YOLO
import json
import numpy as np

def main():
    # Load the trained model
    model = YOLO('models/best_model.pt')
    
    # Perform validation
    results = model.val(data='data1.yaml')
    
    # Save all values as lists
    metrics = {
        'precision': results.box.map50.tolist(),  # Convert NumPy array to list
        'recall': results.box.map.tolist(),       # Convert NumPy array to list
        'mAP_50': results.box.map50.tolist(),     # Convert NumPy array to list
        'mAP_50_95': results.box.map.tolist(),    # Convert NumPy array to list
        'f1_score': results.box.f1.tolist(),      # Convert NumPy array to list
    }
    
    # Save the metrics to a JSON file
    with open('D:/Pg-DAI/CV/car_damage_detection/results/predictions/validation_reports.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
