import torch
import torchvision
from torchvision import transforms, models
import tensorflow as tf
import numpy as np 
from PIL import Image
import os

def load_data(type):

    folders = ["finger_gun", "high_five", "okay", "peace", "thumbs_up"]

    images = []
    labels = []

    resize_tf = transforms.Resize(size=(224, 224))      # Images need to be resized for compatibility with ResNet18

    for folder in folders:

        folder_path = "data/" + type + "/" + folder

        for filename in os.listdir(folder_path):

            filepath = os.path.join(folder_path, filename)

            image = resize_tf(Image.open(filepath))
            image = image.rotate(-90, expand=True)       # Resize order is of opposite order (swapped height and width), needs rotation back

            images.append(image)
            labels.append(folder)

    
    images[10].save("debug.png")

    return np.array(images), np.array(labels)

def build_resnet_model(num_classes):    
    # pretrained model
    model = models.resnet18(pretrained=True)
    # change last layer to match our classes from data
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, num_classes)
    
    return model

# based default values off of the last homework, must hyperparamater tune
def train(model, num_epochs=100, learning_rate=1e-4, batch_size=64):
    # TODO: IMPLEMENT ME
    return 

def predict(image_tensor, class_names):
    # TODO: IMPLEMENT ME
    return

def main():

    images_tr, labels_tr = load_data("train")
    
    # Convert the NumPy array to a TensorFlow tensor
    image_tf = tf.convert_to_tensor(images_tr, dtype=tf.float32)
    
    # Normalize the TensorFlow tensor
    image_tf = image_tf / 255.0
    
    # Convert the TensorFlow tensor to a PyTorch tensor
    image_torch = torch.from_numpy(image_tf.numpy()).permute(0, 3, 1, 2)

    torchvision.utils.save_image(image_torch, "debug2.png")
    
    # Print the shapes of the tensors
    print(f'TensorFlow tensor shape: {image_tf.shape}')
    print(f'PyTorch tensor shape: {image_torch.shape}')
    # -----------------------------------------------------------
    
    # 1. parse data and clean for resnet
    
    
    # 2. build model
    
    
    # 3. train model
    
    
    # 4. test model
    

if __name__ == "__main__":
    main()