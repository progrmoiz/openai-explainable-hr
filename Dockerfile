# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Add this after your base image specification (e.g., FROM python:3.7-slim)
RUN apt-get update && \
    apt-get -y install gcc g++ python3-dev

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --upgrade -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --chdir app app:app