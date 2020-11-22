### Mostly Finished Dockerfile ###

# set base image (host os)
FROM python:3.8

# Not needed
# set the environment variable to the private key user json file
# this environment variable is needed so google functions can access my project
# ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/access.json

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local directoy to the working directory
COPY / .

# command to run on container start
CMD [ "python", "./app.py" ]
