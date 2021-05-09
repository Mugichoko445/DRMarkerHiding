#pragma once

#include <opencv2/aruco.hpp>

#include "ArUcoMarker/Marker.h"

class ArUcoMarker : public Marker
{
public:
	ArUcoMarker(int id, float size, float margin, cv::aruco::PREDEFINED_DICTIONARY_NAME dictionaryName = cv::aruco::DICT_6X6_250);
	~ArUcoMarker();

	void DetectMarkers(cv::InputArray image);
	void GetCorners(cv::OutputArray corners);
	void EstimatePoseSingleMarkers(cv::InputArray cameraMatrix, cv::InputArray distCoeffs);

	void DrawDetectedMarkers(cv::InputArray src, cv::OutputArray dst);
	void DrawAxis(cv::InputArray src, cv::OutputArray dst, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, float axisLength);

private:
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners;
	std::vector<std::vector<cv::Point2f>> rejections;
	cv::Ptr<cv::aruco::DetectorParameters> parameters;
	cv::Ptr<cv::aruco::Dictionary> dictionary;

	std::vector<cv::Vec3d> rvecs, tvecs;
};