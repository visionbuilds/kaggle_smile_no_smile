import torch
from torchvision import models, transforms
from PIL import Image
import os

class Predictor:
    def __init__(self, model_weights_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = torch.load(model_weights_path, map_location=self.device)
        self.model.eval()  # Устанавливаем режим оценки

        # Трансформации для изображения
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict_image(self,image_path):
        try:
            image = Image.open(image_path).convert("RGB")  # Убедимся, что это RGB-изображение
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return

        # Преобразование изображения
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # Добавляем batch dimension

        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        predicted_class = class_names[predicted_idx.item()]
        # print(f"Предсказанный класс: {predicted_class}")
        return predicted_class

    def predict_folder(self,folder_path):
        for file in os.scandir(folder_path):
            predicted_class = self.predict_image(file.path)
            print(file.name,predicted_class)

# Пример использования
if __name__ == "__main__":
    # Замените пути на ваши данные
    # image_path = r"A:\pycharm_projects\idk-ml-restricted-objects\EfficientNet\data\val\non_smile\Abel_Pacheco_0004.jpg"
    model_weights_path = "efficientnet_weights1.pt"
    class_names = ['non_smile', 'smile']
    predictor=Predictor(model_weights_path, class_names)
    # Запуск предсказания
    # predict_image(image_path, model_weights_path, class_names)
    dir=r'A:\pycharm_projects\idk-ml-restricted-objects\EfficientNet\data\val\smile'
    predictor.predict_folder(dir)