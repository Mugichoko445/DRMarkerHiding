#include "PixMixMarkerHiding.h"

namespace dr
{
	PixMixMarkerHiding::PixMixMarkerHiding(const ArUcoMarker& marker, bool debugViz)
		: markerSize(marker.Size()), markerMargin(marker.Margin()), debugViz(debugViz)
	{
	}

	PixMixMarkerHiding::~PixMixMarkerHiding()
	{
	}

	void PixMixMarkerHiding::Reset(cv::InputArray color, cv::InputArray corners, const det::PixMixParams& params)
	{
		assert(corners.cols() > 0);

		std::vector<cv::Point2f> newMkCors;
		AddMarginToMarkerCorners(corners, newMkCors);

		// PixMix
		cv::Mat inpainted, nnf, cost, mask;
		dr::util::CreateMaskFromCorners(newMkCors, color.size(), mask);
		pm.Run(color, mask, inpainted, nnf, cost, params);
		kf.Set(inpainted, mask, nnf, cost, corners);

		std::cout << "[PixMixMarkerHiding::Rest] Inpainted a keyframe" << std::endl;
	}

	void PixMixMarkerHiding::Run(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners, const det::PixMixParams& params)
	{
		if (!kf.IsEmpty() && corners.size() == kf.Corners().size())
		{
			cv::Mat refColor, refNNF, refCost;
			kf.GetWarped(corners, refColor, refNNF, refCost);

			cv::Mat newMkCorns, mask;
			AddMarginToMarkerCorners(corners, newMkCorns);
			dr::util::CreateMaskFromCorners(newMkCorns, color.size(), mask);

			// fill in non-masked area with the original color
			std::random_device rnd;
			auto mt = std::mt19937(rnd());
			auto rRand = std::uniform_int_distribution<int>(0, refColor.rows - 1);
			auto cRand = std::uniform_int_distribution<int>(0, refColor.cols - 1);
			for (int r = 0; r < refColor.rows; ++r)
			{
				auto refColorPtr = refColor.ptr<cv::Vec3b>(r);
				auto colorPtr = color.getMat().ptr<cv::Vec3b>(r);
				auto refNNFPtr = refNNF.ptr<cv::Vec2i>(r);
				auto maskPtr = mask.ptr<uchar>(r);
				for (int c = 0; c < refColor.cols; ++c)
				{
					if (maskPtr[c] != 0)
					{
						refColorPtr[c] = colorPtr[c];
						refNNFPtr[c] = cv::Vec2i(r, c);
					}

					if (refNNFPtr[c][0] < 0 || refNNFPtr[c][0] >= refColor.rows
						|| refNNFPtr[c][1] < 0 || refNNFPtr[c][1] >= refColor.cols)
					{
						refNNFPtr[c][0] = rRand(mt);
						refNNFPtr[c][1] = cRand(mt);
					}
				}
			}

			det::PixMixKeyframe ref;
			ref.Set(refColor, mask, refNNF, refCost, corners);

			if (debugViz)
			{
				cv::imshow("debug - reference color", refColor);
				cv::waitKey(1);
			}

			pm.Run(color, mask, ref, inpainted, params);
		}
	}

	void PixMixMarkerHiding::AddMarginToMarkerCorners(cv::InputArray corners, cv::OutputArray newCorners)
	{
		// Create a mask image with marker margin
		std::vector<cv::Point2f> mkCors(corners.cols());
		mkCors[0] = cv::Point2f(markerSize, 0.0f);
		mkCors[1] = cv::Point2f(markerSize, markerSize);
		mkCors[2] = cv::Point2f(0.0f, markerSize);
		mkCors[3] = cv::Point2f(0.0f, 0.0f);
		cv::Matx33f H = cv::findHomography(mkCors, corners);
		// corners with margin
		std::vector<cv::Point3f> mkCorsWithMarg(mkCors.size());
		mkCorsWithMarg[0] = H * cv::Point3f(mkCors[0].x + markerMargin, mkCors[0].y - markerMargin, 1.0f);
		mkCorsWithMarg[1] = H * cv::Point3f(mkCors[1].x + markerMargin, mkCors[1].y + markerMargin, 1.0f);
		mkCorsWithMarg[2] = H * cv::Point3f(mkCors[2].x - markerMargin, mkCors[2].y + markerMargin, 1.0f);
		mkCorsWithMarg[3] = H * cv::Point3f(mkCors[3].x - markerMargin, mkCors[3].y - markerMargin, 1.0f);
		std::vector<cv::Point2f> newMkCors(mkCors.size());
		for (int idx = 0; idx < mkCorsWithMarg.size(); ++idx)
		{
			newMkCors[idx] = cv::Point2f(mkCorsWithMarg[idx].x, mkCorsWithMarg[idx].y) / mkCorsWithMarg[idx].z;
		}

		cv::Mat res(corners.size(), corners.type(), &newMkCors.front());
		res.copyTo(newCorners);
	}
}