FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile","Pipfile.lock","/app/"]
RUN pipenv install --system --deploy
COPY ["docker_starter.py","/app/"]
ENTRYPOINT ["python", "docker_starter.py"]