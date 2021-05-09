# DR Marker Hiding

This is an open source project of Diminished Reality (DR) I introduced in the "Visualization and Graphics in Mixed Reality" tutorial at [Eurographics 2021](https://conferences.eg.org/eg2021/program/tutorials/). This project would be a good starting point for DR beginners like students. All written in C++, mainly using OpenCV 4.

This project contains three marker hiding methods written in C++ (See below for the details). Here, "marker hiding" means to remove a fiducial marker from a video stream in real time, as if the marker didn't exist in the scene, so that we can mimic markerless Augmented Reality (AR) in 6DoF.

> **_NOTE:_** Since no publicly available original implementations of those DR methods exist, this implementation does not guarantee the quality described in the original papers.

Enjoy! :+1:

[Shohei Mori](https://mugichoko445.github.io), TUGraz, Austria

## Methods
### ```Siltanen```

This implementation is based on the paper ```S. Siltanen, “Texture Generation over the Marker Area,” Proc. ISMAR (2006)```, a pioneering work of DR marker hiding.

### ```PixMixMarkerHiding```

This implementation is based on the paper ```J. Herling and W. Broll, "High Quality Real Time Video Inpainting with PixMix," IEEE TVCG, Vol. 20, Issue 6, pp. 866 - 879, 2014.``` The original implementation uses a feature point-based object tracking, while this implementation relies on the marker detection and tracking. Due to the inaccuracy of the marker detection, you would observe jiggling artifacts within the inpainted area.

### ```MtMarkerHiding```

This implementation is based on the visualization technique presented in ```N. Kawai, T. Sato, and N. Yokoya, "Diminished Reality based on Image Inpainting Considering Background," IEEE TVCG, Vol. 22 Issue 3, pp. 1236 - 1247, 2016.``` This implementation shows ongoing inpainting result, during the inpainting optimization, on each image pyramid level completed in another thread. This visualization technique should be in line with the original implementation. Poisson seamless blending mitigates the color inconsistency between the inpainted area and the vicinity area.

**Note** The scene multi-plane detection in the original paper is not implemented in this example code. Also, note that the original implementation uses a different inpainting method, while this marker hiding application uses a varient of the PixMix inpainting, for which I crunk up more rasterscan iterations and more random sampling during the optimization.


## How to Start?
Everything is implemented in C++, mainly using ```OpenCV 4```. I tested the code on a Windows machine (Windows 10 Pro) and in Visual Studio 2019 Community.


### Installing OpenCV
As an example, here is a simple instraction of how to install ```OpenCV``` using ```vcpkg```, to your Visual Studio projects. Again, note that this is for a Windows x64 build, on which I tested the code.

1. Clone the [vcpkg GitHub repository](https://github.com/microsoft/vcpkg): e.g. by ```git clone https://github.com/microsoft/vcpkg```
2. Move to the cloned directory: ```cd ./vcpkg```
3. Bootstrap the vcpkg: ```.bootstrap-vcpkg.bat```
4. Download and build an ```OpenCV``` package: ```vcpkg install opencv4:x64-windows```
	* This may take a while, e.g., several hours
5. Make the installed package available in your Visual Studio projects: ```vcpkg integrate install```

### Building the Solution

1. Open ```build/DR-Tutorial.sln```
2. Build all the three projects in ```Release (x64)``` mode

### Running the applications

There are three steps: 1) print out, 2) calibration, and 3) marker hiding. 

#### Print Out

1. Print out each page of [Print-This.pdf](Print-This.pdf) in two sheets of A4 paper
2. Cut ArUco markers
	* The marker ID is ```23```
	* There are three markers with the same ID but in different sizes
	* Use one on your preference

#### Camera Calibration

1. Then, run ```bin/x64_Release/CameraCalibration.exe```
	* It runs with a default arguments, but to see the detail run ```bin/x64_Release/CameraCalibration.exe -help```
2. Capture (```c``` key) the checkerboard target in p.1 of [Print-This.pdf](Print-This.pdf)
	* Take approximately 25 images of the chessboard target
3. Start the calibration (```r``` key)
	* ```data/ip.xml``` will be generated

#### DR Marker Hiding

1. Finally, run ```bin/x64_Release/DR-MarkerHiding.exe```
	* It runs with a default arguments, but to see the detail run ```bin/x64_Release/CameraCalibration.exe -help```
	* ```Siltanen```: This method immediately inpaints a marker once the marker is detected
	* ```PixMixMarkerHiding```: Press the ```r``` key to (re-)start inpainting
	* ```MtMarkerHiding```: Press the ```r``` key to start inpainting. While the inpainting progresses, its the ongoing inpainted results are shown on the marker accordingly


_To Be Added_ Here's a video instruction showing how the code should work.
