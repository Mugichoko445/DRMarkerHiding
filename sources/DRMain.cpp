#include <opencv2/highgui.hpp>

#include "ArUcoMarker/ArUcoMarker.h"
#include "DR/Siltanen/Siltanen.h"
#include "DR/PixMix/PixMixMarkerHiding.h"
#include "DR/KawaiViz/MtMarkerHiding.h"
#include "CameraCalibration/Calibration.h"

void RunSiltanen(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs);
void RunPixMixMarkerHiding(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs);
void RunMtMarkerHiding(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs);

int main(int argc, char** argv) try
{
	cv::setUseOptimized(true);

	cv::String keys =
		"{help h||Show help command}"
		"{id|0|USB camera ID}"
		"{xml_name xn|../../data/ip.xml|Input XML file name}"
		"{method m|s|s: Siltanen, p: PixMix, m: Multi-threading}";
	cv::String about = "Copyright Shohei Mori";
	cv::CommandLineParser parser(argc, argv, keys);
	
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	auto cameraID = parser.get<int>("id");
	auto xmlName = parser.get<cv::String>("xml_name");
	auto method = parser.get<cv::String>("method");

	std::cout << "[DRMain] Input summary" << std::endl;
	std::cout << " - Camera ID: " << cameraID << std::endl;
	std::cout << " - Input XML name: " << xmlName << std::endl;
	std::cout << " - Method: " << method << std::endl;

	cv::Size imageSize;
	cv::Mat cameraMatrix, distCoeffs;
	io::ReadIntrinsics(std::string(xmlName), imageSize, cameraMatrix, distCoeffs);

	std::cout << "[DRMain] from XML" << std::endl;
	std::cout << " - Camera image size: " << imageSize << std::endl;
	std::cout << " - Camera matrix: " << cameraMatrix << std::endl;
	std::cout << " - Distortion coefficients: " << distCoeffs << std::endl;

	cv::VideoCapture cam(cameraID);
	if (!cam.isOpened()) std::cerr << "[main] Error opening video stream!" << std::endl;

	cam.set(cv::CAP_PROP_FRAME_WIDTH, imageSize.width);
	cam.set(cv::CAP_PROP_FRAME_HEIGHT, imageSize.height);

	ArUcoMarker marker(23, 0.036f, 0.02f);

	if (method == "s") RunSiltanen(cam, marker, cameraMatrix, distCoeffs);
	else if (method == "p") RunPixMixMarkerHiding(cam, marker, cameraMatrix, distCoeffs);
	else if (method == "m") RunMtMarkerHiding(cam, marker, cameraMatrix, distCoeffs);
	else std::cerr << "[main] Method " << method << " is not found!" << std::endl;

	return 0;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	exit(EXIT_FAILURE);
}


void RunSiltanen(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs)
{
	dr::Siltanen ip(marker, 256, true);

	const std::string wndName("DR View");
	while (true)
	{
		cv::Mat color, inpainted, viz;
		cam >> color;

		std::vector<cv::Point2f> corners;
		marker.DetectMarkers(color);
		marker.EstimatePoseSingleMarkers(cameraMatrix, distCoeffs);
		marker.GetCorners(corners);

		ip.Run(color, inpainted, corners);

		if (!inpainted.empty())
		{
			marker.DrawAxis(inpainted, viz, cameraMatrix, distCoeffs, 0.05f);
			cv::imshow(wndName, viz);
		}
		else cv::imshow(wndName, color);
		if (cv::waitKey(1) == 27 /* escape key */) break;
	}
}

void RunPixMixMarkerHiding(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs)
{
	dr::PixMixMarkerHiding pmMk(marker, true);

	const std::string wndName("DR View");
	char key = -1;
	while (key != 27 /* escape key */)
	{
		cv::Mat color, inpainted, viz;
		cam >> color;

		std::vector<cv::Point2f> corners;
		marker.DetectMarkers(color);
		marker.EstimatePoseSingleMarkers(cameraMatrix, distCoeffs);
		marker.GetCorners(corners);

		// inpainting
		if (corners.size() > 0 && key == 'r' /* r (reset) key*/)
		{
			dr::det::PixMixParams params;
			params.alpha = 0.5f;
			params.maxItr = 10;

			pmMk.Reset(color, corners, params);
		}
		else if (corners.size() > 0 && pmMk.IsInitiated())
		{
			dr::det::PixMixParams params;
			params.alpha = 0.0f;
			params.maxItr = 1;

			pmMk.Run(color, inpainted, corners, params);
		}

		if (!inpainted.empty())
		{
			marker.DrawAxis(inpainted, viz, cameraMatrix, distCoeffs, 0.05f);
		}
		else
		{
			viz = color.clone();
		}
		cv::putText(viz, cv::String("[r] rest, [esc] to exit"), cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255));
		cv::imshow(wndName, viz);

		key = cv::waitKey(1);
	}
}

void RunMtMarkerHiding(cv::VideoCapture& cam, ArUcoMarker& marker, cv::InputArray cameraMatrix, cv::InputArray distCoeffs)
{
	dr::MtMarkerHiding pmMtMk(marker, 128, 768, true);
	
	const std::string wndName("DR View");
	char key = -1;
	while (key != 27 /* escape key */)
	{
		cv::Mat color, inpainted, intermidColor, viz;
		cam >> color;

		std::vector<cv::Point2f> corners;
		marker.DetectMarkers(color);
		marker.EstimatePoseSingleMarkers(cameraMatrix, distCoeffs);
		marker.GetCorners(corners);

		// inpainting
		if (corners.size() > 0 && pmMtMk.IsDone() && key == 'r' /* r (reset) key*/)
		{
			dr::det::PixMixParams params;
			params.alpha = 0.5f;
			params.maxItr = 20;
			params.maxRandSearchItr = 20;
			pmMtMk.Run(color, corners, inpainted, params);
		}

		if (corners.size() > 0 && pmMtMk.GetIntermidColor(color, intermidColor, corners))
		{
			viz = intermidColor.clone();
		}
		else
		{
			viz = color.clone();
		}

		marker.DrawAxis(viz, viz, cameraMatrix, distCoeffs, 0.05f);

		cv::imshow(wndName, viz);
		key = cv::waitKey(1);
	}

	pmMtMk.Stop();
}