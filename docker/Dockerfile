FROM python:3.10-slim

ARG PORT

ENV PORT=${PORT:-5001}

RUN mkdir -p /app/files

COPY . /app

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE $PORT

ENTRYPOINT [ "python" ]

CMD [ "api.py" ]
