#pragma once

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
#include <opencv2/ccalib.hpp>

namespace io
{
	bool SaveIntrinsics(const std::string& filename, const cv::Size& imageSize, cv::InputArray cameraMatrix, cv::InputArray distCoeffs);
	bool ReadIntrinsics(const std::string& filename, cv::Size& imageSize, cv::OutputArray cameraMatrix, cv::OutputArray distCoeffs);
}

class Calibration
{
public:
	Calibration(int gridX, int gridY, float size);
	~Calibration();

	void Run(const cv::Size& imageSize, cv::InputOutputArray cameraMatrix, cv::InputOutputArray distCoeffs);
	bool DetectCorners(cv::InputArray image, cv::OutputArray viz);
	inline int GetDetectionCount() const
	{
		return static_cast<int>(vPts2d.size());
	}

private:
	cv::Size grid;
	float size;
	std::vector<std::vector<cv::Point2f>> vPts2d;
	std::vector<cv::Point3f> pts3d;
};