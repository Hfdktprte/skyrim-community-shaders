#pragma once

namespace NeuralRendering
{
	/** Runs DLSS 5 Neural Rendering on the flat LDR scene immediately before UI composite. */
	bool ApplyLdr();

	/** Releases all runtime and shared-resource state. */
	void Reset();
}
