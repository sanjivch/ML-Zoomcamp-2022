
FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /homework

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["serve_model.py", "model1.bin", "dv.bin", "./"]

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD [ "serve_model.py", "--host=0.0.0.0"]
