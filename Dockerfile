FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.11.8 AS uv
FROM --platform=linux/amd64 ubuntu:22.04
 
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt-get update && \
    apt-get install -y wget git vim tmux sudo build-essential libxrender1 ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
 
# Install uv (for the correct amd64 platform)
COPY --from=uv /uv /usr/local/bin/uv

# Add binaries to path
ENV PATH=/usr/local/bin:$PATH

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Binaries

# reduce
RUN wget https://github.com/rlabduke/reduce/archive/refs/tags/v4.15.tar.gz && \
    tar -xf v4.15.tar.gz && \
    cd reduce-4.15/ && \
    make && \
    cp reduce_src/reduce /usr/local/bin/ && \
    cd .. && rm -r reduce-4.15/ && rm -r v4.15.tar.gz
 
# gnina (note we might want to upgrade to gnina 1.3 but it requires a CUDA installation)
RUN wget https://github.com/gnina/gnina/releases/download/v1.1/gnina \
    -O /usr/local/bin/gnina && chmod +x /usr/local/bin/gnina
