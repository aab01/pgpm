# Inference script for running predictions with the trained model

import os
import torch as T
from model.unet1sc import UNet1SC
from datagenerator import makesignal

def load_model(model_path, device):
    model = UNet1SC(n_channels=1, n_classes=6)
    model.load_state_dict(T.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_data, device):
    with T.no_grad():
        input_tensor = T.tensor(input_data, dtype=T.float32).to(device)
        output = model(input_tensor.unsqueeze(0))  # Add batch dimension
        return output[0].cpu().numpy()

def main():
    device = "cuda" if T.cuda.is_available() else "cpu"
    model_path = './savedmodels/modelFinal.pth'
    
    model = load_model(model_path, device)
    
    # Example of generating a signal for inference
    x_tmp, gt, signalclass = makesignal(noise_level=0.2)
    predictions = run_inference(model, x_tmp, device)
    
    # Here you can add code to visualize or save the predictions

if __name__ == "__main__":
    main()