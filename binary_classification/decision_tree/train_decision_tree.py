import os
from deepface import DeepFace
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from joblib import dump, load
# Путь к папкам с данными
train_dir = r"A:\pycharm_projects\smile_not_smile\EfficientNet\data\train"
valid_dir = r"A:\pycharm_projects\smile_not_smile\EfficientNet\data\valid"

# Классы (имена подпапок)
classes = ["smile", "non_smile"]


# Функция для получения данных и их эмбеддингов
def load_data_and_embeddings(data_dir, model_name="VGG-Face"):
    embeddings = []
    labels = []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Получение эмбеддинга изображения
                embedding = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
                embeddings.append(embedding[0]["embedding"])
                labels.append(label)  # Метка класса (0 для smile, 1 для non_smile)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    return np.array(embeddings), np.array(labels)


# Загрузка обучающих данных
print("Загрузка обучающих данных...")
X_train, y_train = load_data_and_embeddings(train_dir)

# Загрузка валидационных данных
print("Загрузка валидационных данных...")
X_valid, y_valid = load_data_and_embeddings(valid_dir)

# Обучение дерева решений
print("Обучение модели...")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
dump(clf, 'decision_tree.joblib')
# Оценка модели
y_pred = clf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)

print(f"Точность модели на валидационном наборе: {accuracy * 100:.2f}%")

# with PCA
pca = PCA(n_components=256)
X_train_reduced = pca.fit_transform(X_train)
X_valid_reduced = pca.transform(X_valid)

clf = DecisionTreeClassifier()
clf.fit(X_train_reduced, y_train)

# Оценка модели
y_pred = clf.predict(X_valid_reduced)
accuracy = accuracy_score(y_valid, y_pred)

print(f"Точность модели на валидационном наборе: {accuracy * 100:.2f}%")