
# Object_Detection_AngleCalculation_Yolo_CV
Object detection Model with inclined angle calculation feature.\
It incorporates yolov8, opencv2, with custom built yolov8 training dataset.
![alt text](https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV/blob/main/1.jpg?raw=true)
![alt text](https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV/blob/main/2.jpg?raw=true)
![alt text](https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV/blob/main/3.jpg?raw=true)
![alt text](https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV/blob/main/4.jpg?raw=true)
![alt text](https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV/blob/main/5.jpg?raw=true)
## Installation
1. Create a new venv 
```
python venv -m Your_Venv_Name /path/to/your/venv/location
```
2. Activate venv
```
/path/to/your/venv/Scripts/Activate.ps1
```
3. Clone this git repo
```
git clone https://github.com/imranadas/Object_Detection_AngleCalculation_Yolo_CV.git
```
4. Install libraries from requirements.txt in active venv
```
pip install -r /path/to/requirements.txt
```
## Usage/Examples
1. Training Examples: edit train.py and use yolov8 pretrained model or train from strach pre-trained models are preferables for weight transfer and efficient model training.
```
/path/to/venv/Scripts/python.exe /path/to/repo/train.py
```
2. Detection of Books in images: detection fucntions are available in detection.py, use them freely in conjugtion with draw bounding_boxes functions.
3. Detection and Angle Calculation: Run angle_calc.py in active venv and pass PATH in the terminal.
```
/path/to/venv/Scripts/python.exe /path/to/repo/angle_calc.py
```
Proceed detection and calcualtion by pressing enter on the image preview.\
Program will store results locally in CWD/results folder\
Re-Run of program will delete pre-existing result folder in CWD.

## Dependencies
certifi==2022.12.7\
charset-normalizer==2.1.1\
colorama==0.4.6\
contourpy==1.1.1\
cycler==0.12.1\
filelock==3.9.0\
fonttools==4.43.1\
fsspec==2023.4.0\
idna==3.4\
Jinja2==3.1.2\
kiwisolver==1.4.5\
MarkupSafe==2.1.2\
matplotlib==3.8.0\
mpmath==1.3.0\
networkx==3.0\
numpy==1.24.1\
opencv-python==4.8.1.78\
packaging==23.2\
pandas==2.1.1\
Pillow==9.3.0\
psutil==5.9.6\
py-cpuinfo==9.0.0\
pyparsing==3.1.1\
python-dateutil==2.8.2\
pytz==2023.3.post1\
PyYAML==6.0.1\
requests==2.28.1\
scipy==1.11.3\
seaborn==0.13.0\
six==1.16.0\
sympy==1.12\
thop==0.1.1.post2209072238\
torch==2.1.0+cu118\
torchaudio==2.1.0+cu118\
torchvision==0.16.0+cu118\
tqdm==4.66.1\
typing_extensions==4.4.0\
tzdata==2023.3\
ultralytics==8.0.199\
urllib3==1.26.13
## Documentation

[iopenCV2](https://opencv.org/)\
[yoloV8](https://github.com/ultralytics/ultralytics)

