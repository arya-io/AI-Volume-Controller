# Hand Gesture Volume Control System

## Overview
The **Hand Gesture Volume Control System** is a project that enables users to control the system's volume through hand gestures. Utilizing computer vision techniques, this system recognizes specific gestures and translates them into volume control actions. 

## Features
- Real-time hand gesture recognition
- Volume control through simple hand movements
- User-friendly interface

## Technologies Used
- **Python**: The primary programming language for implementation.
- **OpenCV**: A library used for computer vision tasks, specifically to capture and process video input.
- **MediaPipe**: A framework for building multimodal applied machine learning pipelines, used here for hand-tracking.

## Installation
To run this project, you'll need to have Python installed on your machine. Follow these steps to set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/arya-io/AI-Volume-Controller.git
   ```

2. Navigate to the project directory:
   ```bash
   cd AI-Volume-Controller
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python main.py
2. Increase and decrease the distance between the tip of your index finger and thumb to control the volume of the system.

## Contributing
Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- OpenCV for image processing capabilities.
- MediaPipe for efficient hand tracking.
