FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["python", "main.py"]
