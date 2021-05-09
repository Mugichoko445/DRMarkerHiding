#pragma once

#include "PixMix.h"
#include "ArUcoMarker/ArUcoMarker.h"
#include "Utilities.h"

namespace dr
{
	class PixMixMarkerHiding
	{
	public:
		PixMixMarkerHiding(const ArUcoMarker& marker, bool debugViz = false);
		~PixMixMarkerHiding();

		void Reset(cv::InputArray color, cv::InputArray corners, const det::PixMixParams& params);
		void Run(cv::InputArray color, cv::OutputArray inpainted, cv::InputArray corners, const det::PixMixParams& params);
		inline bool const IsInitiated() const { return !kf.IsEmpty(); }

	private:
		PixMix pm;
		det::PixMixKeyframe kf;

		float markerSize, markerMargin;
		bool debugViz;

		void AddMarginToMarkerCorners(cv::InputArray corners, cv::OutputArray newCorners);
	};
}