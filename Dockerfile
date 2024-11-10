# Use an appropriate base image with Java and Python
FROM openjdk:8-jdk-slim

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory into '/app'
COPY . /app/

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
