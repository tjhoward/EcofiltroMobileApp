# Python image to use.
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .
COPY modelXGBoost.h5 .

# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN mkdir -p /app/blobs
# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]


# FROM python:3.7

# #Install dependencies
# RUN pip install Flask gunicorn flask_cors 

# #Copy source code to folder in the container
# #COPY src/ app/

# #set container folder as working directory
# #WORKDIR /app
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./


# ENV PORT 8080

# #run commands
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app