from deepface import DeepFace

# Путь к изображению
image_path = r"A:\Downloads\books\a\fotos\1411314426276.jpg"

# Извлечение face embedding с использованием модели ArcFace и MTCNN для детекции
embedding = DeepFace.represent(
    img_path=image_path,
    model_name="ArcFace",
    detector_backend="mtcnn"
)

print("Face embedding:", embedding)