#include "Calibration.h"

bool io::SaveIntrinsics(const std::string& filename, const cv::Size& imageSize, cv::InputArray cameraMatrix, cv::InputArray distCoeffs)
{
	auto dir = fs::path(filename).parent_path();
	if (fs::create_directories(dir))
	{
		std::cout << "[SaveIntrinsics] Created a directory " << dir << std::endl;
	}

	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened())
	{
		std::cerr << "[SaveIntrinsics] Failed to open " << filename << std::endl;
		return false;
	}

	std::cout << "[SaveIntrinsics] Opened " + filename << std::endl;
	fs << "imageSize" << imageSize;
	fs << "cameraMatrix" << cameraMatrix.getMat();
	fs << "distCoeffs" << distCoeffs.getMat();

	return true;
}

bool io::ReadIntrinsics(const std::string& filename, cv::Size& imageSize, cv::OutputArray cameraMatrix, cv::OutputArray distCoeffs)
{
	if (!fs::exists(filename))
	{
		std::cerr << "[ReadIntrinsics] " << filename << " does not exist!" << std::endl;
		return false;
	}

	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cerr << "[SaveIntrinsics] Failed to open " << filename << std::endl;
		return false;
	}

	std::cout << "[ReadIntrinsics] Opened " + filename << std::endl;
	cv::Mat tmpCameraMatrix, tmpDistCoeffs;
	fs["imageSize"] >> imageSize;
	fs["cameraMatrix"] >> tmpCameraMatrix;
	fs["distCoeffs"] >> tmpDistCoeffs;
	tmpCameraMatrix.copyTo(cameraMatrix);
	tmpDistCoeffs.copyTo(distCoeffs);

	return true;
}



Calibration::Calibration(int gridWidth, int gridHeight, float size) : grid(gridWidth, gridHeight), size(size)
{
	// 3D corners of your checkerboard
	for (int y = 0; y < grid.height; ++y) for (int x = 0; x < grid.width; ++x)
	{
		pts3d.push_back(cv::Point3f(y * this->size, x * this->size, 0.0f));
	}
}

Calibration::~Calibration()
{
	// pass
}

void Calibration::Run(const cv::Size& imageSize, cv::InputOutputArray cameraMatrix, cv::InputOutputArray distCoeffs)
{
	std::vector<std::vector<cv::Point3f>> vPts3d;
	vPts3d.resize(vPts2d.size(), pts3d);

	std::vector<cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(vPts3d, vPts2d, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
}

bool Calibration::DetectCorners(cv::InputArray image, cv::OutputArray viz)
{
	assert(image.type() == CV_8U || image.type() == CV_8UC3);

	cv::Mat gray = image.getMat();
	if (gray.type() == CV_8UC3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> pts2d;
	auto found = cv::findChessboardCorners(gray, grid, pts2d);
	if (found)
	{
		cv::cornerSubPix(gray, pts2d, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
		vPts2d.push_back(pts2d);

		cv::Mat imageViz = image.getMat().clone();
		cv::drawChessboardCorners(imageViz, grid, pts2d, found);
		imageViz.copyTo(viz);
	}

	return found;
}
