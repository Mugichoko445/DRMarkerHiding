#include "ArUcoMarker/ArUcoMarker.h"

ArUcoMarker::ArUcoMarker(int id, float size, float margin, cv::aruco::PREDEFINED_DICTIONARY_NAME dictionaryName) : Marker(id, size, margin)
{
	parameters = cv::aruco::DetectorParameters::create();
	dictionary = cv::aruco::getPredefinedDictionary(dictionaryName);
}

ArUcoMarker::~ArUcoMarker()
{
}

void ArUcoMarker::DetectMarkers(cv::InputArray image)
{
	cv::aruco::detectMarkers(image, dictionary, corners, ids, parameters, rejections);
}

void ArUcoMarker::GetCorners(cv::OutputArray dst)
{
	for (int idx = 0; idx < ids.size(); ++idx)
	{
		if (ids[idx] == ID())
		{
			cv::Mat(corners[idx]).copyTo(dst);
			break;
		}
	}
}

void ArUcoMarker::EstimatePoseSingleMarkers(cv::InputArray cameraMatrix, cv::InputArray distCoeffs)
{
	cv::aruco::estimatePoseSingleMarkers(corners, Size(), cameraMatrix, distCoeffs, rvecs, tvecs);
}

void ArUcoMarker::DrawDetectedMarkers(cv::InputArray src, cv::OutputArray dst)
{
	cv::Mat tmp; src.copyTo(tmp);
	if (ids.size() > 0) cv::aruco::drawDetectedMarkers(tmp, corners, ids);
	tmp.copyTo(dst);
}

void ArUcoMarker::DrawAxis(cv::InputArray src, cv::OutputArray dst, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, float axisLength)
{
	cv::Mat tmp; src.copyTo(tmp);
	for (int idx = 0; idx < rvecs.size(); ++idx)
	{
		if (ids[idx] != ID()) continue;
		
		auto rvec = rvecs[idx];
		auto tvec = tvecs[idx];
		cv::aruco::drawAxis(tmp, cameraMatrix, distCoeffs, rvec, tvec, axisLength);
	}
	tmp.copyTo(dst);
}