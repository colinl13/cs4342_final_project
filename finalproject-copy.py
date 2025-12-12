import torch

if __name__ == "__main__":
    
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))