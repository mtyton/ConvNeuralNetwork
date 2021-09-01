# What is Convolutional Neural Network?
Convolutioanl neural network is a special type of neural network which allows to classify images.
It's called by this algorithm main operation "convolution". 
Convolution is a mathematical operation I won't get much into it here, if you are intrested
You can read more here: [LINK](https://en.wikipedia.org/wiki/Convolution)

# What this convolutional neural network does?
This code is prepared to recognize images of pets, exactly dogs and cats.

# Setting up the project
To run it you have to clone this repository. After that you should download data from [HERE](https://drive.google.com/file/d/1l9gQ6kh1dGpm46EPKXFNmKgSgFmjjEvf/view?usp=sharing)
Next you should unpack this archive in the same dir as cloned repository exists.
Your next step should be creating virtual environment, if you are not familliar with this idea, you can read more here: [LINK](https://docs.python.org/3/library/venv.html)
After creation and activation of virtual environment you have to install required packages:
```pip install -r requirements.txt```
should do the job.

# Running neural network

## Training
First you should run training option, to train your neural network:
``` 
python main.py train [<int>number_of_iterations]
```
* number_of_iterations - defaultly set to 10, this parameter should be an integer, but it is not required

## Testing trained network
After your network has been trained, you may test it:
```
python main.py test --it
```
* --it - with this parameter test becomes interactive, and program will display it's guesses. Otherwise it'll display just the final score.
