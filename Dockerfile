FROM python:3.10-bullseye

# Встановлюємо системні залежності для OpenCV та Mediapipe
RUN apt-get update && apt-get install -y \
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

# Оновлюємо pip і встановлюємо Python-пакети
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копіюємо весь проєкт
COPY . .

# Відкриваємо порт для Streamlit
EXPOSE 8501

# Команда запуску
CMD ["streamlit", "run", "pages/Main.py", "--server.address=0.0.0.0", "--server.port=8501"]
