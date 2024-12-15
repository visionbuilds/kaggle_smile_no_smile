import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Устройство (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pass
pass
# Пути к данным
data_dir = "./data"  # Папка с подкаталогами "dogs", "cats", "bears"

# Гиперпараметры
batch_size = 6
num_epochs = 10
learning_rate = 0.001
num_classes = 3

# Трансформации для данных
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Датасеты
image_datasets = {
    "train": datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms["train"]),
    "val": datasets.ImageFolder(root=f"{data_dir}/val", transform=data_transforms["val"]),
}

# Загрузчики данных
dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
    "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False),
}

# Размеры датасетов
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
print(class_names)
# Загрузка модели EfficientNet
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Функция обучения
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Режим обучения
            else:
                model.eval()  # Режим оценки

            running_loss = 0.0
            running_corrects = 0

            # Итерация по данным
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Обнуление градиентов
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Обратный проход и оптимизация только в режиме обучения
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

# Запуск обучения
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

# Сохранение модели
torch.save(model.state_dict(), "efficientnet_weights1.pth")

print("Обучение завершено. Модель сохранена.")
