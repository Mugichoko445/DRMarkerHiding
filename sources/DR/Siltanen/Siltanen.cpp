#include "DR/Siltanen.h"

namespace dr
{
	Siltanen::Siltanen(const Marker& marker, int markerSizeInPx, bool debugViz) : debugViz(debugViz)
	{
		const auto pxRatio = markerSizeInPx / marker.Size();
		const auto marginInPx = int(marker.Margin() * pxRatio);
		const auto vicinitySizeInPx = markerSizeInPx / 2;
		const auto ipImageSizeInPx = (markerSizeInPx + marginInPx) * 2;

		markerRect.x = markerRect.y = vicinitySizeInPx + marginInPx;
		markerRect.width = markerRect.height = markerSizeInPx;
		roiRect.x = roiRect.y = vicinitySizeInPx;
		roiRect.width = roiRect.height = markerSizeInPx + marginInPx * 2;

		ipImage = cv::Mat(ipImageSizeInPx, ipImageSizeInPx, CV_8UC3);

		markerCorners.reserve(4);
		markerCorners.push_back(cv::Point2i(markerRect.x + markerRect.width, markerRect.y));
		markerCorners.push_back(cv::Point2i(markerRect.x + markerRect.width, markerRect.y + markerRect.height));
		markerCorners.push_back(cv::Point2i(markerRect.x, markerRect.y + markerRect.height));
		markerCorners.push_back(cv::Point2i(markerRect.x, markerRect.y));

		roiCorners.reserve(4); // [note] take points at 1px off towards the outside of the ROI
		roiCorners.push_back(cv::Point2i(roiRect.x + roiRect.width, roiRect.y - 1)); // x0, y0
		roiCorners.push_back(cv::Point2i(roiRect.x + roiRect.width, roiRect.y + roiRect.height));  // x1, y1
		roiCorners.push_back(cv::Point2i(roiRect.x - 1, roiRect.y + roiRect.height)); // x2, y2
		roiCorners.push_back(cv::Point2i(roiRect.x - 1, roiRect.y - 1)); // x3, y3

		roiCornersF.reserve(roiCorners.size());
		for (const auto& pt : roiCorners) roiCornersF.push_back(cv::Point2f(pt));

		zigZagLUT.reserve(roiRect.x * 2);
		for (int idx = 0; idx < roiRect.x; ++idx) zigZagLUT.push_back(idx);
		for (int idx = roiRect.x - 1; idx >= 0; --idx) zigZagLUT.push_back(idx);
	}

	Siltanen::~Siltanen()
	{
	}

	void Siltanen::Run(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners)
	{
		if (corners.cols() != markerCorners.size()) return;

		// warp
		auto H = cv::findHomography(corners, markerCorners);
		cv::warpPerspective(color, ipImage, H, ipImage.size());

		// inpaint
		const auto c0 = ipImage.at<cv::Vec3b>(roiCorners[0]);
		const auto c1 = ipImage.at<cv::Vec3b>(roiCorners[1]);
		const auto c2 = ipImage.at<cv::Vec3b>(roiCorners[2]);
		const auto c3 = ipImage.at<cv::Vec3b>(roiCorners[3]);
		for (int r = 0; r < roiRect.height; ++r)
		{
			auto y4 = zigZagLUT[(r + roiRect.y) % zigZagLUT.size()];
			auto y5 = zigZagLUT[(roiRect.height - 1 - r) % zigZagLUT.size()] + roiCorners[1].y;

			auto c4Ptr = ipImage.ptr<cv::Vec3b>(y4);
			auto c5Ptr = ipImage.ptr<cv::Vec3b>(y5);
			auto cPtr = ipImage.ptr<cv::Vec3b>(r + roiRect.y);
			for (int s = 0; s < roiRect.width; ++s)
			{
				auto x6 = zigZagLUT[(roiRect.width - 1 - s) % zigZagLUT.size()] + roiCorners[0].x;
				auto x7 = zigZagLUT[(s + roiRect.x) % zigZagLUT.size()];

				const auto c4 = c4Ptr[s + roiRect.x];
				const auto c5 = c5Ptr[s + roiRect.x];
				const auto c6 = cPtr[x6];
				const auto c7 = cPtr[x7];

				cPtr[s + roiRect.x] = Blend(c0, c1, c2, c3, c4, c5, c6, c7, float(r), float(s), float(roiRect.width));;
			}
		}

#pragma region DEBUG_VIZ
		if (debugViz)
		{
			auto debugImage = ipImage.clone();
			for (const auto& pt : roiCorners)
			{
				cv::circle(debugImage, pt, 1, cv::Scalar(255, 128, 0));
				cv::circle(debugImage, pt, 5, cv::Scalar(255, 128, 0));
			}
			cv::imshow("debug - warped image", debugImage);
			cv::waitKey(1);
		}
#pragma endregion

		// warp back
		cv::Mat dst = color.getMat().clone();
		cv::warpPerspective(ipImage, dst, H.inv(), color.size());

		// composition
		std::vector<cv::Point2f> transRoiCornersF;
		cv::perspectiveTransform(roiCornersF, transRoiCornersF, H.inv());
		std::vector<cv::Point2i> transRoiCornersI;
		transRoiCornersI.reserve(transRoiCornersF.size());
		for (const auto& pt : transRoiCornersF) transRoiCornersI.push_back(cv::Point2i(pt));

		cv::Mat mask(color.size(), CV_8U, cv::Scalar(255));
		cv::fillConvexPoly(mask, transRoiCornersI, cv::Scalar(0));

		auto src = color.getMat();
		for (int y = 0; y < mask.rows; ++y)
		{
			auto mPtr = mask.ptr<uchar>(y);
			auto srcPtr = src.ptr<cv::Vec3b>(y);
			auto dstPtr = dst.ptr<cv::Vec3b>(y);
			for (int x = 0; x < mask.cols; ++x)
			{
				if (mPtr[x] == 255) dstPtr[x] = srcPtr[x];
			}
		}

		// copy
		dst.copyTo(inpainted);
	}
}