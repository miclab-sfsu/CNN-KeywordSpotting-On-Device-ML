
# README for Keyword Spotting Project

## Introduction
The Keyword Spotting project is a real-time system for recognizing specific spoken words or phrases. Utilizing deep learning, it analyzes audio inputs to identify and classify spoken keywords, suitable for applications such as voice-activated assistants and home automation systems.

## Quick Start

### Hardware Requirements
- **Sony Spresense main board**: The core microcontroller unit for processing.
- **Spresense extension board**: For additional connectivity and features.
- **Microphone**: Following the [tutorial for using multiple microphone inputs with Spresense](https://developer.sony.com/spresense/tutorials-sample-projects/spresense-tutorials/using-multiple-microphone-inputs-with-spresense).
- **OLED Display (SSD1306)**: For real-time display of keyword spotting results.
- **32GB Micro SD Card**: Used to store the DSP driver for audio processing.

### Software Dependencies
While `environment_install.sh` takes care of setting up the software environment, it is important to be aware of the key dependencies:
- Python 3.8 or later
- PyTorch
- ONNX
- NumPy
- Additional Python libraries listed in requirements.txt

### How to Install Dependencies
Install the required Python libraries by executing the following command in your terminal:

```bash
bash ./environment_install.sh
````

### Preparing the Dataset
Download the dataset from this [link](https://drive.google.com/file/d/1-EhZPkG6cQYGEU76XD5L0g_h-68mJH1B/view?usp=sharing) and unzip it in the project's root folder. Ensure that the data directory structure is correct as expected by the scripts in the project.


### How to Run the Project
Execute the `run.sh` script to run the entire project, which includes steps for training, testing, and model conversion:

```bash
bash ./run.sh
````

### How to Deploy on Arduino
The deployment on Arduino is facilitated through the `run.sh` script, which prepares the model in a compatible format:

1. Run the `run.sh` script to ensure the model is converted to a TensorFlow Lite format and then to a C header file.
2. Integrate the generated C header file with your Arduino sketch and upload it to the Arduino environment.

