from deepface import DeepFace

# Сравнение двух изображений
path0=r"A:\pycharm_projects\smile_not_smile\EfficientNet\data\train\non_smile\Bob_Stoops_0004.jpg"
path1=r"A:\pycharm_projects\smile_not_smile\EfficientNet\data\train\non_smile\Bob_Stoops_0007.jpg"
result = DeepFace.verify(path0,path1)
print("Are the images of the same person?", result["verified"])