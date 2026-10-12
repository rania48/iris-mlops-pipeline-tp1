FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p models

CMD ["sh", "-c", "python src/train.py && python src/predict.py && python src/drift.py"]