# Используем образ python-slim
FROM python:3.9-slim

# Установливаем зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


# Рабочая директория в контейнере
WORKDIR /app

# Установливаем библиотеки и созданём окружение Python
COPY requirements.txt /app/
RUN python -m venv app/venv && \
    pip install --no-cache-dir -r /app/requirements.txt

# Путь к интерпритатору в вирутальном окружении
ENV PATH="/app/venv/bin:$PATH"

# Копируем все файлы в контейнер
COPY . /app/

# Порт, на котором будет работать приложение
EXPOSE 8080

# Запуск приложения
CMD ["python", "main.py"]
