FROM python:3.10

WORKDIR /app

COPY ./api /app
COPY ./gcp_lib /app/gcp_lib
COPY ./data-scraping /app/data-scraping
COPY ./models /app/models
COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
