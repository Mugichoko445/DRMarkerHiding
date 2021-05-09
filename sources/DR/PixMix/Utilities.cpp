#include "DR/PixMix/Utilities.h"

namespace dr
{
	namespace util
	{
		void CreateMaskFromCorners(cv::InputArray corners, const cv::Size& size, cv::OutputArray mask)
		{
			cv::Mat convexCorners = corners.getMat().clone();
			if (convexCorners.type() == CV_32FC2) convexCorners.convertTo(convexCorners, CV_32SC2);

			cv::Mat maskImg(size, CV_8U, cv::Scalar(255));
			cv::fillConvexPoly(maskImg, convexCorners, cv::Scalar(0));
			maskImg.copyTo(mask);
		}

		void CreateVizPosMap(const cv::InputArray srcPosMap, cv::OutputArray dstColorMap)
		{
			auto src = cv::Mat2i(srcPosMap.getMat());
			auto dst = cv::Mat3b(srcPosMap.size());

			for (int r = 0; r < src.rows; ++r) for (int c = 0; c < src.cols; ++c)
			{
				dst(r, c)[0] = int((float)src(r, c)[1] / (float)src.cols * 255.0f);
				dst(r, c)[1] = int((float)src(r, c)[0] / (float)src.rows * 255.0f);
				dst(r, c)[2] = 255;
			}

			dst.copyTo(dstColorMap);
		}
	}
}