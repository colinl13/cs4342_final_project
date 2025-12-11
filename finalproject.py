import torch
import tf 
import numpy as np 
from PIL import Image

def main():
    # Load a sample image using PIL
    image_path = 'sample_image.jpg'
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    image_np = np.array(image)
    
    # Convert the NumPy array to a TensorFlow tensor
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    
    # Normalize the TensorFlow tensor
    image_tf = image_tf / 255.0
    
    # Convert the TensorFlow tensor to a PyTorch tensor
    image_torch = torch.from_numpy(image_tf.numpy()).permute(2, 0, 1)  # Change from HWC to CHW format
    
    # Print the shapes of the tensors
    print(f'PIL Image size: {image.size}')
    print(f'NumPy array shape: {image_np.shape}')
    print(f'TensorFlow tensor shape: {image_tf.shape}')
    print(f'PyTorch tensor shape: {image_torch.shape}')