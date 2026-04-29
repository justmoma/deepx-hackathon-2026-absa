FROM python:3.10-slim

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Start the Flask app using Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
