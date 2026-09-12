#include "Integration.h"

#include "Renderer.h"
#include "Features/HDRDisplay.h"
#include "Features/Upscaling.h"
#include "Globals.h"

namespace NeuralRendering
{
	namespace
	{
		eastl::unique_ptr<Texture2D> color;
		std::uint32_t colorWidth = 0;
		std::uint32_t colorHeight = 0;
		DXGI_FORMAT colorFormat = DXGI_FORMAT_UNKNOWN;
		std::uint32_t lastAppliedFrame = UINT32_MAX;
		bool routeWasActive = false;
		bool hdrBlockLogged = false;
		bool vrBlockLogged = false;

		ID3D11Texture2D* ResolveRenderTargetTexture(
			const RE::BSGraphics::RenderTargetData& target,
			winrt::com_ptr<ID3D11Texture2D>& holder)
		{
			if (target.texture)
				return target.texture;
			auto resolveView = [&](ID3D11View* view) -> ID3D11Texture2D* {
				if (!view)
					return nullptr;
				winrt::com_ptr<ID3D11Resource> resource;
				view->GetResource(resource.put());
				if (!resource || FAILED(resource->QueryInterface(holder.put())))
					return nullptr;
				return holder.get();
			};
			if (auto* texture = resolveView(target.SRV))
				return texture;
			return resolveView(target.RTV);
		}

		bool EnsureColorResource(ID3D11Resource* source, std::uint32_t width, std::uint32_t height)
		{
			winrt::com_ptr<ID3D11Texture2D> sourceTexture;
			if (!source || FAILED(source->QueryInterface(sourceTexture.put())))
				return false;
			D3D11_TEXTURE2D_DESC sourceDesc{};
			sourceTexture->GetDesc(&sourceDesc);
			if (color && colorWidth == width && colorHeight == height && colorFormat == sourceDesc.Format)
				return true;
			color = Upscaling::CreateTextureFromSource(source, width, height, false, true, true,
				"NeuralRendering::LdrColor");
			if (!color)
				return false;
			colorWidth = width;
			colorHeight = height;
			colorFormat = sourceDesc.Format;
			return true;
		}

		Tuning GetTuning(const Upscaling::Settings& settings)
		{
			return {
				settings.neuralRenderingIntensity,
				settings.neuralRenderingLocalTone,
				settings.neuralRenderingLocalStructure,
				settings.neuralRenderingGlobalTone,
				settings.neuralRenderingSkinStructure,
				settings.neuralRenderingResolutionScale * 0.01f,
				settings.neuralRenderingPassCount,
				settings.neuralRenderingStyle,
				settings.neuralRenderingAutoMask,
				settings.neuralRenderingUICorrection,
			};
		}
	}

	bool ApplyLdr()
	{
		auto& upscaling = globals::features::upscaling;
		const bool hdrConfigured = globals::features::hdrDisplay.loaded &&
			globals::features::hdrDisplay.settings.enableHDR;
		const bool routeActive = !REL::Module::IsVR() &&
			upscaling.GetUpscaleMethod() == Upscaling::UpscaleMethod::kDLSS &&
			upscaling.settings.neuralRenderingEnabled && !hdrConfigured;
		if (!routeActive) {
			if (upscaling.settings.neuralRenderingEnabled && REL::Module::IsVR() && !vrBlockLogged) {
				logger::warn("[DLSSNR] Standalone Neural Rendering currently supports flat Skyrim only");
				vrBlockLogged = true;
			}
			if (upscaling.settings.neuralRenderingEnabled && hdrConfigured && !hdrBlockLogged) {
				logger::warn("[DLSSNR] LDR route blocked: disable HDR Display");
				hdrBlockLogged = true;
			}
			if (routeWasActive)
				Reset();
			routeWasActive = false;
			return false;
		}
		routeWasActive = true;

		const std::uint32_t frame = globals::state ? globals::state->frameCount : 0;
		if (lastAppliedFrame == frame)
			return true;

		auto* renderer = globals::game::renderer;
		auto* context = globals::d3d::context;
		if (!renderer || !context || !globals::d3d::device || !upscaling.motionVectorCopyTexture)
			return false;

		winrt::com_ptr<ID3D11Texture2D> framebufferHolder;
		auto& target = renderer->GetRuntimeData().renderTargets[RE::RENDER_TARGETS::kFRAMEBUFFER];
		auto* framebuffer = ResolveRenderTargetTexture(target, framebufferHolder);
		auto& depth = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN];
		if (!framebuffer || !depth.texture || !depth.depthSRV || !upscaling.motionVectorCopyTexture->resource)
			return false;

		D3D11_TEXTURE2D_DESC colorDesc{};
		D3D11_TEXTURE2D_DESC motionDesc{};
		framebuffer->GetDesc(&colorDesc);
		upscaling.motionVectorCopyTexture->resource->GetDesc(&motionDesc);
		if (!EnsureColorResource(framebuffer, colorDesc.Width, colorDesc.Height))
			return false;
		const auto guideContentWidth = std::clamp(static_cast<std::uint32_t>(
			std::lround(static_cast<double>(motionDesc.Width) * upscaling.resolutionScale.x)), 1u, motionDesc.Width);
		const auto guideContentHeight = std::clamp(static_cast<std::uint32_t>(
			std::lround(static_cast<double>(motionDesc.Height) * upscaling.resolutionScale.y)), 1u, motionDesc.Height);

		ID3D11RenderTargetView* savedRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{};
		ID3D11DepthStencilView* savedDSV = nullptr;
		context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, savedRTVs, &savedDSV);
		context->OMSetRenderTargets(0, nullptr, nullptr);
		context->CopyResource(color->resource.get(), framebuffer);

		ID3D12Device* sharedDevice = nullptr;
		ID3D12CommandQueue* sharedQueue = nullptr;
		if (upscaling.IsFrameGenerationDx12PathActive()) {
			sharedDevice = upscaling.dx12SwapChain.d3d12Device.get();
			sharedQueue = upscaling.dx12SwapChain.commandQueue.get();
		}

		const bool succeeded = Renderer::Instance().Apply(globals::d3d::device, context, 0,
			color->resource.get(), depth.texture, depth.depthSRV,
			upscaling.motionVectorCopyTexture->resource.get(), motionDesc.Width, motionDesc.Height,
			guideContentWidth, guideContentHeight, colorDesc.Width, colorDesc.Height, static_cast<float>(motionDesc.Width),
			static_cast<float>(motionDesc.Height), GetTuning(upscaling.settings),
			sharedDevice, sharedQueue);
		if (succeeded) {
			context->CopyResource(framebuffer, color->resource.get());
			lastAppliedFrame = frame;
		}

		context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, savedRTVs, savedDSV);
		for (auto*& rtv : savedRTVs)
			if (rtv) rtv->Release();
		if (savedDSV) savedDSV->Release();
		return succeeded;
	}

	void Reset()
	{
		Renderer::Instance().Reset();
		color.reset();
		colorWidth = 0;
		colorHeight = 0;
		colorFormat = DXGI_FORMAT_UNKNOWN;
		lastAppliedFrame = UINT32_MAX;
		routeWasActive = false;
	}
}
