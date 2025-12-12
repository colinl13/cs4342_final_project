import torch
import torchvision
from torchvision import transforms, models
from torchvision.transforms import functional
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np 
import os
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import MulticlassAccuracy

def load_data(type):

    folders = ["finger_gun", "high_five", "okay", "peace", "thumbs_up"]

    images = []
    labels = []

    resize_tf = transforms.Resize(size=(224, 224))      # Images need to be resized for compatibility with ResNet18
    normalize_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for label_idx, folder in enumerate(folders):

        folder_path = "data/" + type + "/" + folder

        for filename in os.listdir(folder_path):

            filepath = os.path.join(folder_path, filename)

            image = resize_tf(torchvision.io.read_image(filepath))
            image = image.float() / 255.0
            image = normalize_tf(image)
            image = functional.rotate(image, -90)

            labels.append(label_idx)
            images.append(image)

    image_tensors = torch.stack(images)

    return image_tensors, torch.from_numpy(np.array(labels))

def build_resnet_model(num_classes):    
    # pretrained model
    model = models.resnet18(pretrained=True)
    # change last layer to match our classes from data
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, num_classes)
    
    return model

# based default values off of the last homework, must hyperparamater tune
def train(images, labels, model, num_epochs=100, learning_rate=1e-4, batch_size=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        for i in range(0, len(images), batch_size):
            
            inputs = images[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

# model is resnet model
# test_loader is DataLoader object with test data
# device is the device being run on (defined in train)
def test(model, images, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5

    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)

    model.eval()

    with torch.no_grad():
        for images, labels in images, labels:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            accuracy.update(preds,labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
    
    acc_per_class = accuracy.compute().cpu().numpy()
    prec_per_class = precision.compute().cpu().numpy()
    rec_per_class = recall.compute().cpu().numpy()
    f1_per_class = f1.compute().cpu().numpy()

    return acc_per_class, prec_per_class, rec_per_class, f1_per_class

#def visualize_g_x(model, )

def predict(image_tensor, class_names):
    return

def main():

    images_tr, labels_tr = load_data("train")

    torchvision.utils.save_image(images_tr, "debug2.png")
    
    # Print the shapes of the tensors
    print(f'PyTorch tensor shape: {images_tr.shape}')

    model = build_resnet_model(5)

    train(images_tr, labels_tr, model)

    images_te, labels_te = load_data("test")
    
    test(model, images_te, labels_te)


if __name__ == "__main__":

    main()