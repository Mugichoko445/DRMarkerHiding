#pragma once

#include <thread>
#include <mutex>
#include <opencv2/core.hpp>
#include "ArUcoMarker/Marker.h"
#include "DR/PixMix/PixMix.h"

namespace dr
{
	class MtMarkerHiding
	{
	public:
		MtMarkerHiding(const Marker& marker, int markerSizeInPx, int maxIpImageSize, bool debugViz);
		~MtMarkerHiding();

		void Run(cv::InputArray color, cv::InputArray corners, cv::OutputArray inpainted, const det::PixMixParams& params);
		bool GetIntermidColor(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners);
		void Stop();
		bool IsDone();

	private:
		PixMix pm;
		std::thread th;
		cv::Mat intermidColor;

		cv::Mat ipColor, ipMask;
		std::vector<cv::Point2i> markerCorners;
		std::vector<cv::Point2i> roiCorners;
		std::vector<cv::Point2f> roiCornersF;
		cv::Rect markerRect, roiRect;

		bool debugViz;
	};
}