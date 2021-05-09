#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include "OneLvPixMix.h"

namespace dr
{
	namespace det
	{
		class PixMixKeyframe
		{
		public:
			void Set(cv::InputArray color, cv::InputArray mask, cv::InputArray nnf, cv::InputArray cost, cv::InputArrayOfArrays corners);
			const void GetWarped(cv::InputArray corners, cv::OutputArray warpedColor, cv::OutputArray warpedNNF, cv::OutputArray warpedCost);

			inline const bool IsEmpty() const { return color.empty(); }
			inline const cv::Mat& Color() const { return color; }
			inline const cv::Mat& Mask() const { return mask; }
			inline const cv::Mat& NNF() const { return nnf; }
			inline const cv::Mat& Cost() const { return cost; }
			inline const cv::Mat& Corners() const { return corners; }

		private:
			cv::Mat color, mask, nnf, cost, corners;
		};
	}

	class PixMix
	{
	public:
		PixMix();
		~PixMix();

		void Run(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, cv::OutputArray nnf, cv::OutputArray cost, const det::PixMixParams& params, bool debugViz = false);
		void Run(cv::InputArray color, cv::InputArray mask, const det::PixMixKeyframe& ref, cv::OutputArray inpainted, const det::PixMixParams& params);
		
	private:
		std::vector<det::OneLvPixMix> pm;

		void BuildPyrm(cv::InputArray color, cv::InputArray mask, const int maxPyrmLv);
		int CalcPyrmLv(int width, int height, int maxPyrmLv);
		void FillInLowerLv(det::OneLvPixMix& pmUpper, det::OneLvPixMix& pmLower);
		void BlendBorder(cv::InputArray color, cv::InputArray mask, cv::OutputArray dst, int blurSize);

#pragma region MULTITHREADING
	public:
		void MtRun(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, const det::PixMixParams& params);
		bool GetIntermidColor(cv::OutputArray color);
		void StopMt();
		bool IsDone();

	private:
		std::thread th;
		std::atomic<bool> terminate, done;
		static std::mutex copyMtx;
		cv::Mat intermidColor, mtColor, mtMask, mtNNF, mtCost;	// for MtMarkerHiding
#pragma endregion
	};
}