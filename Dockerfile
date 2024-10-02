# Use a base image with Python
FROM python:3.10

# Install the repository directly from GitHub
RUN pip install git+https://github.com/lauracabayol/TEMPERATURE_FORECAST.git

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY app/requirements.txt .

# Install necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of your application files into the container
COPY app/ .

# Expose the port the app runs on (if needed)
EXPOSE 9999

# Command to run your Jupyter notebook (you can change this as necessary)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=9999", "--no-browser", "--allow-root", "--NotebookApp.token=12345"]


