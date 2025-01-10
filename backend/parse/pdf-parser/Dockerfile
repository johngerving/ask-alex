FROM python:3.10.12

RUN mkdir /usr/src/app

COPY pika_credentials.py consumer.py /usr/src/app

COPY requirements.txt /usr/src/app

WORKDIR /usr/src/app

RUN pip install docling pika python-dotenv requests
CMD ["python", "./consumer.py"]