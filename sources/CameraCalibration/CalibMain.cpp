#include <iostream>
#include <opencv2/highgui.hpp>
#include "Calibration.h"


int main(int argc, char** argv) try
{
	cv::setUseOptimized(true);

	const cv::String keys =
		"{help||Show help command}"
		"{id|0|USB camera ID}"
		"{width w|640|Camera image width}"
		"{height h|480|Camera image height}"
		"{size s|35|Marker rectangle size (mm)}"
		"{grid_w gw|9|The number of rectangle corners in X direction}"
		"{grid_h gh|7|The number of rectangle corners in Y direction}"
		"{xml_name xn|../../data/ip.xml|Output XML file name}";
	const cv::String about = "Copyright Shohei Mori";
	cv::CommandLineParser parser(argc, argv, keys);

	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	auto cameraID = parser.get<int>("id");
	auto width = parser.get<int>("width");
	auto height = parser.get<int>("height");
	auto size = parser.get<float>("size");
	auto gridW = parser.get<int>("grid_w");
	auto gridH = parser.get<int>("grid_h");
	auto xmlName = parser.get<cv::String>("xml_name");

	std::cout << "[CalibMain] Input summary" << std::endl;
	std::cout << " - Camera ID: " << cameraID << std::endl;
	std::cout << " - Width: " << width << std::endl;
	std::cout << " - Height: " << height << std::endl;
	std::cout << " - Size of checkerboard rectangle (mm): " << size << std::endl;
	std::cout << " - Checkerboard grids: " << cv::Size(gridW, gridH) << std::endl;
	std::cout << " - Output XML name: " << xmlName << std::endl;

	cv::VideoCapture cam(cameraID);
	if (!cam.isOpened()) throw "[main] Error opening video stream";
	
	cam.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cam.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	Calibration calib(gridW, gridH, size);

	char key = -1;
	while (key != 27 /* escape key */)
	{
		cv::Mat frame, vizFrame, vizCorners;
		cam >> frame;
		
		vizFrame = frame.clone();
		cv::putText(vizFrame, cv::String("[c] capture, [r] calibration, [esc] exit"), cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255));
		cv::imshow("Camera frame", vizFrame);
		key = cv::waitKey(1);

		if (key == 'c')
		{
			auto found = calib.DetectCorners(frame, vizCorners);
			if (found)
			{
				auto msg = cv::String("Found " + std::to_string(calib.GetDetectionCount()) + " checkerboard(s)");
				cv::putText(vizCorners, msg, cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255));
				cv::imshow("Detected corners", vizCorners);
				cv::waitKey(1);
			}
		}
		else if (key == 'r')
		{
			cv::Mat cameraMatrix, distCoeffs;
			std::cout << "[CalibMain] Start calibration! It may take a while..." << std::endl;
			calib.Run(frame.size(), cameraMatrix, distCoeffs);
			io::SaveIntrinsics(xmlName, frame.size(), cameraMatrix, distCoeffs);
			std::cout << "[CalibMain] done!" << std::endl;
		}
	}

	return 0;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	exit(EXIT_FAILURE);
}