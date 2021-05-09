#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include "ArUcoMarker/Marker.h"

namespace dr
{
	class Siltanen
	{
	public:
		Siltanen(const Marker& marker, int markerSizeInPx = 256, bool debugViz = false);
		~Siltanen();

		void Run(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners);

	private:
		cv::Mat ipImage;
		
		std::vector<cv::Point2i> markerCorners;
		std::vector<cv::Point2i> roiCorners;
		std::vector<cv::Point2f> roiCornersF;
		cv::Rect markerRect, roiRect;

		std::vector<int> zigZagLUT;
		
		bool debugViz;

		inline const cv::Vec3b Blend(
			const cv::Vec3i& c0, const cv::Vec3i& c1, const cv::Vec3i& c2, const cv::Vec3i& c3,
			const cv::Vec3i& c4, const cv::Vec3i& c5, const cv::Vec3i& c6, const cv::Vec3i& c7,
			float r, float s, float l) const
		{
			return (-r * s / l / l) * c0 + (-r * (l - s) / l / l) * c1
				+ (-(l - r) * (l - s) / l / l) * c2 + (-(l - r) * s / l / l) * c3
				+ s / l * c4 + (l - s) / l * c5 + r / l * c6 + (l - r) / l * c7;
		}
	};
}