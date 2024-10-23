
FROM python:3.7

ENV PYTHONUNBUFFERED True

EXPOSE 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt
RUN python3.7 -m pip install --upgrade pip

CMD streamlit run --server.port 8080 final_app.py
