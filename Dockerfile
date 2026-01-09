FROM continuumio/miniconda3:24.7.1-0

# Create a conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate finalprojectenv" >> ~/.bashrc
ENV PATH="$PATH:/opt/conda/envs/finalprojectenv/bin"

# Create a non-root user and switch to that user
RUN useradd -m finalprojectuser
USER finalprojectuser

# Set working directory, and copy files
WORKDIR /home/finalprojectuser

# Expose the JupyterLab and Dask ports
EXPOSE 8888
EXPOSE 8787

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0"]
