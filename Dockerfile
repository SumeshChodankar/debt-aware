FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH="/app"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
