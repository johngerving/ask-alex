FROM python:3.10.12

RUN mkdir /usr/src/app

COPY main.py docling_converter.py pdf_parser.py /usr/src/app/

WORKDIR /usr/src/app

RUN pip install docling pika python-dotenv requests
RUN pip install -e 'git+https://github.com/johngerving/ask-alex#egg=ask_alex_utils&subdirectory=backend/utils'
CMD ["python", "./consumer.py"]