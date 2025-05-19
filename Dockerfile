FROM python:3.9-slim

# Set working dir in
WORKDIR /app

# Install system packages ( needed for the project )
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential python3-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

#  Upgrade pip and install Cornac’s build deps first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      Cython \
      "numpy<2.0.0" \
      "scipy<=1.13.1" \
      tqdm \
      powerlaw

#  Now install the rest of your requirements, including recommenders
RUN pip install --no-cache-dir -r requirements.txt

# Copy in your FastAPI code -> <project-foldier> of docker  
COPY app/   ./ 

# copy data & model into the container of docker root so PROJECT_ROOT.parent /dataset works
COPY dataset/ /dataset/
COPY model/   /model/

# Expose port and launch
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
