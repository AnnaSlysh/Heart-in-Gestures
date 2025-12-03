FROM python:3.10-slim

# Системні пакети для OpenCV + Mediapipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1 \
    && apt-get clean

# Робоча директорія
WORKDIR /app

# Копіюємо залежності
COPY requirements.txt .

# Встановлюємо залежності
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копіюємо весь проєкт
COPY . .

# Streamlit запускається по порту 8501
EXPOSE 8501

# Команда запуску
CMD ["streamlit", "run", "pages/Main.py", "--server.address=0.0.0.0", "--server.port=8501"]
