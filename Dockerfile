FROM tensorflow/tensorflow

WORKDIR /app

COPY requirements.txt requirements.txt

COPY src src

COPY logs logs

COPY data data

RUN pip install -r requirements.txt

RUN python src/download_stopwords.py

CMD ["python", "src/start_server.py"]

EXPOSE 5095