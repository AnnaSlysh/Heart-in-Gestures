# Базовий образ Python 3.10
FROM python:3.10-bullseye

# Встановлюємо системні бібліотеки для OpenCV, Mediapipe і TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1 \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && apt-get clean

# Робоча директорія
WORKDIR /app

# Копіюємо залежності
COPY requirements.txt .

# Оновлюємо pip та встановлюємо Python-пакети
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копіюємо весь проєкт
COPY . .

# Відкриваємо порт 8080 для сервера
EXPOSE 8080

# Команда запуску Streamlit
CMD ["streamlit", "run", "pages/Main.py", "--server.address=0.0.0.0", "--server.port=8080"]
