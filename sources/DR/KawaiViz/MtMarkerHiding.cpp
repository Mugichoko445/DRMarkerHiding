#include "MtMarkerHiding.h"

namespace dr
{
	MtMarkerHiding::MtMarkerHiding(const Marker& marker, int markerSizeInPx, int maxIpImageSize, bool debugViz) : debugViz(debugViz)
	{
		const auto pxRatio = markerSizeInPx / marker.Size();
		const auto marginInPx = int(marker.Margin() * pxRatio);
		const auto vicinitySizeInPx = (maxIpImageSize - markerSizeInPx - 2 * marginInPx) / 2;
		const auto ipImageSizeInPx = markerSizeInPx + (vicinitySizeInPx + marginInPx) * 2;

		markerRect.x = markerRect.y = vicinitySizeInPx + marginInPx;
		markerRect.width = markerRect.height = markerSizeInPx;
		roiRect.x = roiRect.y = vicinitySizeInPx;
		roiRect.width = roiRect.height = markerSizeInPx + marginInPx * 2;

		ipColor = cv::Mat(ipImageSizeInPx, ipImageSizeInPx, CV_8UC3);

		markerCorners.reserve(4);
		markerCorners.push_back(cv::Point2i(markerRect.x + markerRect.width, markerRect.y));
		markerCorners.push_back(cv::Point2i(markerRect.x + markerRect.width, markerRect.y + markerRect.height));
		markerCorners.push_back(cv::Point2i(markerRect.x, markerRect.y + markerRect.height));
		markerCorners.push_back(cv::Point2i(markerRect.x, markerRect.y));

		roiCorners.reserve(4);
		roiCorners.push_back(cv::Point2i(roiRect.x + roiRect.width - 1, roiRect.y));
		roiCorners.push_back(cv::Point2i(roiRect.x + roiRect.width - 1, roiRect.y + roiRect.height - 1));
		roiCorners.push_back(cv::Point2i(roiRect.x, roiRect.y + roiRect.height - 1));
		roiCorners.push_back(cv::Point2i(roiRect.x, roiRect.y));

		roiCornersF.reserve(roiCorners.size());
		for (const auto& pt : roiCorners) roiCornersF.push_back(cv::Point2f(pt));
	}

	MtMarkerHiding::~MtMarkerHiding()
	{
	}

	void MtMarkerHiding::Run(cv::InputArray color, cv::InputArray corners, cv::OutputArray inpainted, const det::PixMixParams& params)
	{
		if (corners.cols() != markerCorners.size()) return;

		// warp
		auto H = cv::findHomography(corners, markerCorners);
		cv::warpPerspective(color, ipColor, H, ipColor.size());

		dr::util::CreateMaskFromCorners(roiCorners, ipColor.size(), ipMask);

		// start multi-threading
		pm.MtRun(ipColor, ipMask, inpainted, params);
	}

	bool MtMarkerHiding::GetIntermidColor(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners)
	{
		if (corners.cols() != markerCorners.size()) return false;

		cv::Mat intermidColor;
		if (!pm.GetIntermidColor(intermidColor)) return false;

		if (debugViz)
		{
			cv::imshow("debug - inpainting result", intermidColor);
			cv::waitKey(1);
		}

		// warp back
		auto H = cv::findHomography(markerCorners, corners);
		cv::warpPerspective(intermidColor, intermidColor, H, color.size());

		// composition (Poisson seamless cloning)
		std::vector<cv::Point2f> transRoiCornersF;
		cv::perspectiveTransform(roiCornersF, transRoiCornersF, H);
		std::vector<cv::Point2i> transRoiCornersI;
		transRoiCornersI.reserve(transRoiCornersF.size());
		for (const auto& pt : transRoiCornersF) transRoiCornersI.push_back(cv::Point2i(pt));

		cv::Mat mask(color.size(), CV_8U, cv::Scalar(0));
		cv::fillConvexPoly(mask, transRoiCornersI, cv::Scalar(255));

		cv::Point center(0, 0);
		for (const auto& pt : transRoiCornersI) center += pt;
		center = center / int(transRoiCornersI.size());
		cv::seamlessClone(intermidColor, color, mask, center, inpainted, cv::NORMAL_CLONE);

		return true;
	}

	void MtMarkerHiding::Stop()
	{
		pm.StopMt();
	}

	bool MtMarkerHiding::IsDone()
	{
		return pm.IsDone();
	}
}