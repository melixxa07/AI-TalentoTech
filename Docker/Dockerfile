
# We have to indicate all the steps that we want to execute in the container

# 1. Define the base image of the container (In this case is Python). Usamos o creamos una imagen base 
FROM python:3.8-slim-buster
## <<slim-buster>> reduces the image size, making your container more efficient, but it also means that some of the tools you might expect to be available in the image are not there.
## This is a lightweight Python 3.8 image 

#MAINTAINER Melissa
## This is the name of the person who created the Dockerfile

# 2. Set the working directory in the container. It will create this folder so the following commands will be excecuted there
WORKDIR /app

# 3. Copy the dependencies file to the working directory. We need to copy the requirements.txt file or libraries to the container
COPY requirements.txt .

# 4. Install any needed packages or dependencies specified in requirements.txt. We need to install the libraries that are in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install matplotlib==3.3.4
## We can install any library that is not in the requirements.txt file or we can add it to the requirements.txt file directly

# 5. Copy the source code to the working directory. We need to copy the source code to the container
COPY . .
## This will take everything from the current directory and copy it to the working directory in the container
## If we wamt to ignire any file we can use a .dockerignore file and put the name of the files that we want to ignore

# 6. Run app.py when the container launches. We need to run the main file of the application
#CMD ["python", "app.py"]
CMD ["python", "train_model.py"]
## We can run any file that we want to run when the container launches 
## "train_model.py" is going to be run by default when the container launches if I don't specify any other file



#TODO: To build the image we need to run the following command in the terminal:
    # >> docker build -t tensorflow-python:latest .
        ## -t is the tag of the image. In this case is "tensorflow-python" and the version is "latest"
        ## . Tells Docker to use the Dockerfile in the current directory
        ### NOW I HAVE A CUSTOM DOCKER IMAGE CALLED "tensorflow-python:latest"

        #->> docker build --no-cache -t tensorflow-train:latest .

#TODO: To run the container we need to run the following command in the terminal:
    # >> docker run --rm tensorflow-python:latest
        ## --rm is to remove the container when it stops running
        ## tensorflow-python:latest is the name of the image that we want to run

        #->> docker run --rm -v ${PWD}:/app tensorflow-train:latest
            ## -v is to mount the current directory to the /app directory in the container
            ## $(pwd) is to get the current directory
            ## tensorflow-train:latest is the name of the image that we want to run


# -v ${PWD}:/app makes or connects /app inside the container actually to my local project directory (the folder where I'm running Docker). 
# This is useful because it means that I can make changes to my code on my local machine and they will be reflected inside the container.   
# This is also useful because any file II write to /app inside the container will actually be saved to my local folder. 
# So I can see the saved files or images from the code on my local machine.