## Sudoku Solver

This is a Sudoku solver program written in Python 3.
The program has been created as a project during summer semester of the Python in the Enterprise course
at AGH in Cracow.

The project uses OpenCV Python library for the image capture, segmentation, enhancement and processing part.

Example image segmented by the image segmentation module:

[Segmented image](https://drive.google.com/open?id=1cKEywTSBbH8hSuDosLwpGaM_801fLHwi "Segmented image")


Digit recognition part has been created using [Keras API](https://keras.io/ "Keras API Documentation") with [Tensorflow](https://www.tensorflow.org/ "Tensorflow page") backend and is realised
as a 2D convolutional neural network ( CNN ). 

To train the network example [data set](https://drive.google.com/open?id=19WWg2E-NvJ4v0f1SEvj1SihVsiVyVVbK "Train digits data set") has been used. This data set consists of *.csv files that represent 440 different fonts ( for digits only ).
The above font data was downloaded from [University of California archive](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images) website.

The example trained network that is attached in the main repository folder achieved 97.5% of accuracy
( the test set consisted of 10% of the whole data set, and was not used during training, according to the validation rules ) .

[Training accuracy plot](https://drive.google.com/open?id=1MhmXfo-NzXbYHimTLJT9273sFdbslIQC)

Finally the Sudoku solver module has been written in C as a CPython Extension module.
The repository folder **solver/** contains precompiled modules for MS Windows working with Python version 3.6 and Linux working with Python versions 3.4, 3.5 and 3.6 .
Inside the **solver/** directory there are also project source files for the solver module that can be compiled under MS Windows ( using Visual Studio ) and Linux ( with gcc ).
Mac OS has not been tested, but the solver compilation process could possibly work using the gcc.

# Demo

A demo video file showing how the OCR Sudoku solver works can be viewed here:

[Sudoku solver in action](https://drive.google.com/open?id=1Y8--qQIg5aF-AB7V7U96fGcGT8ESZG9j "Video")

The test sudoku has been designed to work against the backtracking algorithm ( is good for benchmarking ).

![Hard sudoku](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Sudoku_puzzle_hard_for_brute_force.svg/361px-Sudoku_puzzle_hard_for_brute_force.svg.png)

You can also visit the [Sudoku algorithms Wikipedia page](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms)

# Installation
Debian / Ubuntu 
To install modules required to run the project, 
create new virtual environment ( or run the commands below directly in the terminal )

``` sudo apt install python3-pip ```    ( if pip for Python 3 is not installed on Your system)

then run

``` pip3 install opencv-python keras tensorflow pandas matplotlib ```

# How to run the solver

To start the solver just run the *`Capture.py`* script from terminal or use IDE such as PyCharm
``` python3 Capture.py ```

# Testing without a video capture device

To test the image recognition capabilities of the solver run

``` python3 ImageProcessor.py ```

there is a image file *rand.jpg* supplied in the repository main folder,
which is used as an input to the digit recognition module.

Sudoku depicted on *rand.jpg* test image is designed to work against the brute force
backtracking algorithm. It is a good benchmark for the solver module written in C
used by the program. Solver is compiled as a CPython extension module.

