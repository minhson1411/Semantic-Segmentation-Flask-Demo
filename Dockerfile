FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the content of the local src directory to the working directory
COPY . .


RUN apt-get update &&\
    pip install -r requirements.txt

EXPOSE 5000

#command to run on container start
ENTRYPOINT ["python3"]
CMD ["app.py" ]
