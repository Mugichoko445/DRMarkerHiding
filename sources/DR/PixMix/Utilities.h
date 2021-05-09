#pragma once

#include <opencv2/opencv.hpp>

namespace dr
{
	namespace util
	{
		void CreateMaskFromCorners(cv::InputArray corners, const cv::Size& size, cv::OutputArray mask);
		void CreateVizPosMap(cv::InputArray srcPosMap, cv::OutputArray dstColorMap);
	}
}