# Use the specific platform version
FROM --platform=linux/amd64 ubuntu:22.04

# Update and install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    libcurl4-openssl-dev \
    psmisc \
    python3-pip \
    unzip \
    vim \
    wget \
    zip \
    openjdk-11-jdk \
    && apt-get clean

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set the MonetDB version to be installed
ENV MONETDB_VERSION=11.49.11

# Download MonetDB packages
RUN wget -q https://www.monetdb.org/downloads/deb/archive/jammy/monetdb5-sql_${MONETDB_VERSION}_amd64.deb \
    && wget -q https://www.monetdb.org/downloads/deb/archive/jammy/monetdb-client_${MONETDB_VERSION}_amd64.deb \
    && wget -q https://www.monetdb.org/downloads/deb/archive/jammy/monetdb5-server_${MONETDB_VERSION}_amd64.deb \
    && wget -q https://www.monetdb.org/downloads/deb/archive/jammy/libmonetdb-client26_${MONETDB_VERSION}_amd64.deb \
    && wget -q https://www.monetdb.org/downloads/deb/archive/jammy/libmonetdb-stream26_${MONETDB_VERSION}_amd64.deb \
    && wget -q https://www.monetdb.org/downloads/deb/archive/jammy/libmonetdb28_${MONETDB_VERSION}_amd64.deb

# Install MonetDB packages and clean up
RUN dpkg -i libmonetdb-stream26_${MONETDB_VERSION}_amd64.deb \
    && dpkg -i libmonetdb-client26_${MONETDB_VERSION}_amd64.deb \
    && dpkg -i libmonetdb28_${MONETDB_VERSION}_amd64.deb \
    && dpkg -i monetdb5-server_${MONETDB_VERSION}_amd64.deb \
    && dpkg -i monetdb5-sql_${MONETDB_VERSION}_amd64.deb \
    && dpkg -i monetdb-client_${MONETDB_VERSION}_amd64.deb \
    && rm -f monetdb5-sql_${MONETDB_VERSION}_amd64.deb \
    && rm -f monetdb-client_${MONETDB_VERSION}_amd64.deb \
    && rm -f monetdb5-server_${MONETDB_VERSION}_amd64.deb \
    && rm -f libmonetdb-client26_${MONETDB_VERSION}_amd64.deb \
    && rm -f libmonetdb-stream26_${MONETDB_VERSION}_amd64.deb \
    && rm -f libmonetdb28_${MONETDB_VERSION}_amd64.deb

# Copy the content of the DfAnalyzer directory to the Docker image
COPY DfAnalyzer/ /opt/DfAnalyzer/

# Set the working directory and ensure script permissions
WORKDIR /opt/DfAnalyzer
RUN chmod +x restore-database.sh

# Run the restore-database.sh script
RUN ./restore-database.sh

COPY example /opt/example/

# Install pytest
RUN pip3 install pytest==7.3.1

# Copy the 'dfa-lib-python' directory to the root of the Docker image
COPY library/dfa-lib-python /opt/dfa-lib-python/


# Install the Python package
WORKDIR /opt/dfa-lib-python/
RUN python3 setup.py install

# Expose necessary ports
EXPOSE 22000 50000

WORKDIR /opt/DfAnalyzer

# Set the default command to start the database and shell
CMD ["/bin/bash"]