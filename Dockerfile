FROM python:3.9-slim

WORKDIR /tg_app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch
RUN pip install redis
RUN pip install fastapi
RUN pip install aiogram
RUN pip install transformers
RUN pip install accelerate
COPY . .

CMD ["python", "bot.py"]
