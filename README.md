# Hand Signal Car

A small C++ / OpenCV program. The webcam watches your left arm, and a top-down car follows the three driving hand signals:

- Arm straight out to the side: the car turns left
- Elbow bent, forearm pointing up: the car turns right
- Elbow bent, forearm pointing down: the car stops
- Arm down at your side: the car drives straight

Stand a few feet back so your shoulders and left hand are in frame. The preview is mirrored, so your left arm is on the left side of the window. The yellow overlay is the arm the program is reading.

## Build

From the project folder, in PowerShell:

```powershell
.\scripts\setup.ps1
.\scripts\build.ps1
.\build\hand_signal_car.exe
```

`setup.ps1` downloads portable CMake, the OpenCV 4.11 Windows libraries, and a YOLOv8n-pose model. The first build uses the Visual Studio 2022 C++ tools already installed on this machine.

## Keys

- `V` returns to the camera
- `A` / `D` force a left or right turn
- `S` stops
- `W` drives straight
- `Q` quits
