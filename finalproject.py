import torch
import torchvision
from torchvision import transforms, models
from torchvision.transforms import functional
import numpy as np 
import os
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import ResNet18_Weights

def load_data(type):

    folders = ["finger_gun", "high_five", "okay", "peace", "thumbs_up"]

    images = []
    labels = []

    # Images need to be resized and normalized for compatibility with ResNet18
    resize_tf = transforms.Resize(size=(224, 224))
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
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # change last layer to match our classes from data
    
    # TEMP
    for param in model.parameters():
        param.requires_grad = False
    
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, num_classes)
    
    return model

# based default values off of the last homework, must hyperparamater tune
# Good at all but finger gun (unfrozen layers): epoch=25 rate=1e-4 batch_size = 10
def train(images, labels, model, num_epochs=2000, learning_rate=2e-3, batch_size=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

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
def test(model, images, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5

    accuracy = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)

    model.eval()

    with torch.no_grad():

        inputs = images.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        print("True dist: ", torch.bincount(labels, minlength=num_classes))
        print("Pred dist: ", torch.bincount(preds, minlength=num_classes))

        accuracy.update(preds, labels)
    
    acc_per_class = accuracy.compute().cpu().numpy()

    return acc_per_class

#def visualize_g_x(model, )

def main():

    images_tr, labels_tr = load_data("train")

    torchvision.utils.save_image(images_tr, "debug_train.png")
    
    # Print the shapes of the tensors
    print(f'PyTorch tensor shape: {images_tr.shape}')

    model = build_resnet_model(5)

    train(images_tr, labels_tr, model)

    images_te, labels_te = load_data("test")
    
    torchvision.utils.save_image(images_te, "debug_test.png")

    print(f"Testing accuracy = {test(model, images_te, labels_te)}")

if __name__ == "__main__":

    main()