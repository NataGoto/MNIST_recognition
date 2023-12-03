import sys
sys.path.insert(0, 'https://github.com/NataGoto/my_first_app1/blob/main/builder.py')
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

# Теперь можно импортировать другие модули, включая builder.py


def process(image_file):
    MODEL_NAME = 'model_fmr_all.h5'  # Убедитесь, что путь к модели указан правильно
    model = load_model(MODEL_NAME)
    INPUT_SHAPE = (28, 28, 1) # традиционная форма МНИСТ 

    image = Image.open(image_file).convert('L')  # Преобразование в черно-белое
    resized_image = image.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
    array = np.array(resized_image)[np.newaxis, ..., np.newaxis]
    array = array / 255.0  # Нормализация

    prediction = model.predict(array)
    predicted_class = np.argmax(prediction, axis=1)  # Получение класса с наивысшей вероятностью
    return resized_image, predicted_class

# Код приложения Streamlit
st.title('MNIST Classification Demo')
image_file = st.file_uploader('Load an image', type=['png', 'jpg'])

if image_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(image_file)
    resized_image, predicted_class = process(image_file)
    col1.text('Source image')
    col1.image(resized_image)
    col2.text('Predicted Class')
    col2.write(predicted_class[0])  # Отображение предсказанного класса
