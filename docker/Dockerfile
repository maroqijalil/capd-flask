FROM python:3.7.11-stretch

ARG PORT

ENV PORT=${PORT:-5001}

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE $PORT

ENTRYPOINT [ "python" ]

CMD [ "api.py" ]
