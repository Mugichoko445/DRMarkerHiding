#include "DR/PixMix/PixMix.h"

namespace dr
{
	namespace det
	{
		void PixMixKeyframe::Set(cv::InputArray color, cv::InputArray mask, cv::InputArray nnf, cv::InputArray cost, cv::InputArrayOfArrays corners)
		{
			this->color = color.getMat().clone();	// inpainted color
			this->mask = mask.getMat().clone();
			this->nnf = nnf.getMat().clone();
			this->cost = cost.getMat().clone();
			this->corners = corners.getMat().clone();
		}

		const void PixMixKeyframe::GetWarped(cv::InputArray corners, cv::OutputArray warpedColor, cv::OutputArray warpedNNF, cv::OutputArray warpedCost)
		{
			cv::Matx33f H = cv::findHomography(this->corners, corners);

			cv::Mat tmpNNF;
			cv::warpPerspective(color, warpedColor, H, color.size(), cv::INTER_LINEAR);
			cv::warpPerspective(nnf, tmpNNF, H, color.size(), cv::INTER_NEAREST);
			cv::warpPerspective(cost, warpedCost, H, color.size(), cv::INTER_NEAREST);

			// warp each pixel position in "nnf"
			for (int r = 0; r < tmpNNF.rows; ++r)
			{
				auto nnfPtr = tmpNNF.ptr<cv::Vec2i>(r);
				for (int c = 0; c < tmpNNF.cols; ++c)
				{
					auto pt = H * cv::Point3f(float(nnfPtr[c][1]), float(nnfPtr[c][0]), 1.0f);
					nnfPtr[c][0] = int(pt.y / pt.z + 0.5f);
					nnfPtr[c][1] = int(pt.x / pt.z + 0.5f);
				}
			}

			tmpNNF.copyTo(warpedNNF);
		}
	}

	PixMix::PixMix() : terminate(false), done(true) { }
	PixMix::~PixMix() { }

	void PixMix::Run(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, cv::OutputArray nnf, cv::OutputArray cost, const det::PixMixParams& params, bool debugViz)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		done.store(false);

		copyMtx.lock();
		std::cout << "[PixMix::Run] Inpainting has started!" << std::endl;
		color.copyTo(intermidColor);
		copyMtx.unlock();

		auto tmpParams = params;
		BuildPyrm(color, mask, tmpParams.maxPyrmLv);

		for (int lv = int(pm.size()) - 1; lv >= 0 && !terminate.load(); --lv)
		{
			if (lv == 0) tmpParams.maxItr = std::min(tmpParams.maxItr, 2);

			pm[lv].Run(tmpParams);
			if (lv > 0) FillInLowerLv(pm[lv], pm[lv - 1]);

			copyMtx.lock();
			cv::resize(*pm[lv].GetColorPtr(), intermidColor, color.size(), 0.0f, 0.0f, cv::INTER_LINEAR);
			copyMtx.unlock();

#pragma region DEBUG_VIZ
			if (debugViz)
			{
				cv::Mat vizColor, vizPosMap;
				cv::resize(*pm[lv].GetColorPtr(), vizColor, color.size(), 0.0f, 0.0f, cv::INTER_NEAREST);
				util::CreateVizPosMap(*pm[lv].GetPosMapPtr(), vizPosMap);
				cv::resize(vizPosMap, vizPosMap, color.size(), 0.0f, 0.0f, cv::INTER_NEAREST);
				cv::imshow("debug - inpainted color", vizColor);
				cv::imshow("debug - colord position map", vizPosMap);
				cv::waitKey(1);
			}
#pragma endregion
		}

		BlendBorder(color, mask, inpainted, tmpParams.blurSize);
		copyMtx.lock();
		inpainted.copyTo(intermidColor);
		copyMtx.unlock();

		pm[0].GetPosMapPtr()->copyTo(nnf);
		pm[0].GetCostMapPtr()->copyTo(cost);

		done.store(true);

		copyMtx.lock();
		std::cout << "[PixMix::Run] Finished the inpainting!" << std::endl;
		copyMtx.unlock();
	}

	void PixMix::Run(cv::InputArray color, cv::InputArray mask, const det::PixMixKeyframe& ref, cv::OutputArray inpainted, const det::PixMixParams& params)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		ref.Color().copyTo(*pm[0].GetColorPtr());
		ref.NNF().copyTo(*pm[0].GetPosMapPtr());
		ref.Cost().copyTo(*pm[0].GetCostMapPtr());

		pm[0].Run(params);

		BlendBorder(color, mask, inpainted, params.blurSize);
	}

	void PixMix::BuildPyrm(cv::InputArray color, cv::InputArray mask, const int maxPyrmLv)
	{
		pm.resize(CalcPyrmLv(color.cols(), color.rows(), maxPyrmLv));
		pm[0].Init(color.getMat(), mask.getMat());
		for (int lv = 1; lv < pm.size(); ++lv)
		{
			auto lvSize = pm[lv - 1].GetColorPtr()->size() / 2;

			// color
			cv::Mat3b tmpColor;
			cv::resize(*(pm[lv - 1].GetColorPtr()), tmpColor, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			// mask
			cv::Mat1b tmpMask;
			cv::resize(*(pm[lv - 1].GetMaskPtr()), tmpMask, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			for (int r = 0; r < tmpMask.rows; ++r)
			{
				auto ptrMask = tmpMask.ptr<uchar>(r);
				for (int c = 0; c < tmpMask.cols; ++c)
				{
					ptrMask[c] = ptrMask[c] < 255 ? 0 : 255;
				}
			}

			pm[lv].Init(tmpColor, tmpMask);
		}
	}

	int PixMix::CalcPyrmLv(int width, int height, int maxPyrmLv)
	{
		auto pyrmLv = 1;
		auto size = std::min(width, height);
		while ((size /= 2) >= 5) ++pyrmLv;

		return std::min(pyrmLv, maxPyrmLv);
	}

	void PixMix::FillInLowerLv(det::OneLvPixMix& pmUpper, det::OneLvPixMix& pmLower)
	{
		cv::Mat colorUpsampled;
		cv::resize(*(pmUpper.GetColorPtr()), colorUpsampled, pmLower.GetColorPtr()->size(), 0.0, 0.0, cv::INTER_LINEAR);
		cv::Mat posMapUpsampled;
		cv::resize(*(pmUpper.GetPosMapPtr()), posMapUpsampled, pmLower.GetPosMapPtr()->size(), 0.0, 0.0, cv::INTER_NEAREST);
		for (int r = 0; r < posMapUpsampled.rows; ++r)
		{
			auto ptr = posMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < posMapUpsampled.cols; ++c) ptr[c] = ptr[c] * 2 + cv::Vec2i(r % 2, c % 2);
		}

		auto colorLw = *(pmLower.GetColorPtr());
		auto maskLw = *(pmLower.GetMaskPtr());
		auto posMapLw = *(pmLower.GetPosMapPtr());

		const int wLw = pmLower.GetColorPtr()->cols;
		const int hLw = pmLower.GetColorPtr()->rows;
		for (int r = 0; r < hLw; ++r)
		{
			auto ptrColorLw = colorLw.ptr<cv::Vec3b>(r);
			auto ptrColorUpsampled = colorUpsampled.ptr<cv::Vec3b>(r);
			auto ptrMaskLw = maskLw.ptr<uchar>(r);
			auto ptrPosMapLw = posMapLw.ptr<cv::Vec2i>(r);
			auto ptrPosMapUpsampled = posMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < wLw; ++c)
			{
				if (ptrMaskLw[c] == 0)
				{
					ptrColorLw[c] = ptrColorUpsampled[c];
					ptrPosMapLw[c] = ptrPosMapUpsampled[c];
				}
			}
		}
	}

	void PixMix::BlendBorder(cv::InputArray color, cv::InputArray mask, cv::OutputArray dst, int blurSize)
	{
		cv::Mat  alphaMask;
		cv::blur(mask, alphaMask, cv::Size(blurSize, blurSize));

		cv::Mat3f colorF, ipColorF, dstColorF(pm[0].GetColorPtr()->size());
		color.getMat().convertTo(colorF, CV_32FC3, 1.0 / 255.0);
		pm[0].GetColorPtr()->convertTo(ipColorF, CV_32FC3, 1.0 / 255.0);

		cv::Mat1f mAlphaF;
		alphaMask.convertTo(mAlphaF, CV_32F, 1.0 / 255.0);

		for (int r = 0; r < color.rows(); ++r)
		{
			auto ptrSrc = colorF.ptr<cv::Vec3f>(r);
			auto ptrPM = ipColorF.ptr<cv::Vec3f>(r);
			auto ptrDst = dstColorF.ptr<cv::Vec3f>(r);
			auto ptrAlpha = mAlphaF.ptr<float>(r);
			for (int c = 0; c < color.cols(); ++c)
			{
				ptrDst[c] = ptrAlpha[c] * ptrSrc[c] + (1.0f - ptrAlpha[c]) * ptrPM[c];
			}
		}

		dstColorF.convertTo(dst, CV_8UC3, 255.0);
	}


#pragma region MULTITHREADING
	std::mutex PixMix::copyMtx;

	void PixMix::MtRun(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, const det::PixMixParams& params)
	{
		if (done.load())
		{
			StopMt();

			// keep the color and mask to make them accessible for PixMix anytime
			color.copyTo(mtColor); mask.copyTo(mtMask);

			th = std::thread([=] { Run(mtColor, mtMask, inpainted, mtNNF, mtCost, params, false); });
		}
	}

	bool PixMix::GetIntermidColor(cv::OutputArray color)
	{
		copyMtx.lock();
		if (!intermidColor.empty()) intermidColor.copyTo(color);
		copyMtx.unlock();

		return !intermidColor.empty();
	}

	void PixMix::StopMt()
	{
		std::cout << "[PixMix::StopMt] Joining the thread..." << std::endl;
		terminate.store(true);
		if (th.joinable()) th.join();
		terminate.store(false);
		std::cout << "[PixMix::StopMt] The thread has been joined!" << std::endl;
	}

	bool PixMix::IsDone()
	{
		return done.load();
	}
#pragma endregion
}