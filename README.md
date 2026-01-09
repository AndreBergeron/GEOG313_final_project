# GEOG313_final_project:
This repository contains all of the final project code Andre Bergeron contributed for the Advanced Geospatial Analytics with Python Course at Clark University

# Utility:
This tool can accessed in the final_report.ipynb jupyter notebook, which can only be run via python programming. The workspace for the code to run our tool is available via a containerized Docker environment, which conatins all the necessary dependencies for users to interactively manipulate and visualize the data.

Users should follow these guidelines to properly set up and run the containerized environment.

- This repository must first be cloned onto a local machine to gain access to the apporiate workspace environment.

- To build the Docker Image, the user should run:
```
docker build -t geog313-final-project .
```

- To run the Docker container, the user should run:
```
docker run -it -p 8888:8888 -p 8787:8787 -v $(pwd):/home/finalprojectuser geog313_final_project
```

- Once the container has been successfully run, the user should attach the container to VScode and select the python environment name "finalprojectenv" in the /home/finalprojectuser directory.

The backend code for this tool is available within the main function, which contains all of the necessary functions to compute the final dashboard.