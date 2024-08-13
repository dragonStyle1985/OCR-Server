# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# 设置源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 创建 /etc/apt/sources.list 文件并替换 Debian 源为阿里云源
RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 4101 available to the world outside this container
EXPOSE 4101

# Define environment variable
ENV NAME Search-Server

ENV EXPIRY_DATE="2025-01-01 11:30"

RUN echo '#!/bin/bash\n\
current_date=$(date +"%Y-%m-%d %H:%M")\n\
echo "Current date and time in container: $current_date"\n\
expiry_date="$EXPIRY_DATE"\n\
# echo "Expiry date and time: $expiry_date"\n\
if [[ "$current_date" > "$expiry_date" ]]; then\n\
    echo "This container has expired. Exiting..."\n\
    exit 1\n\
else\n\
    echo "This container is still valid."\n\
fi\n\
exec "$@"' > /usr/local/bin/expiry_check.sh && \
    chmod +x /usr/local/bin/expiry_check.sh

# Run app.py when the container launches
CMD ["/usr/local/bin/expiry_check.sh", "python3", "app.py"]
