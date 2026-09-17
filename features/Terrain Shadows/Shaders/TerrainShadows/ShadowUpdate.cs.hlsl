Texture2D<float> TexHeight : register(t0);
RWTexture2D<float2> RWTexShadowHeights : register(u0);

cbuffer ShadowUpdateCB : register(b0)
{
	float2 LightPxDir : packoffset(c0.x);   // direction on which light descends, from one pixel to next via dda
	float2 LightDeltaZ : packoffset(c0.z);  // per lightUVDir, normalised, [upper, lower] penumbra, should be negative
	uint StartPxCoord : packoffset(c1.x);
	float2 PxSize : packoffset(c1.y);
	float pad : packoffset(c1.w);           // keep the 1.2.0 DLL constant-buffer ABI unchanged
	float2 PosRange : packoffset(c2.x);
	float2 ZRange : packoffset(c2.z);
}

float GetInterpolatedHeight(float2 pxCoord, bool isVertical)
{
	uint2 dims;
	TexHeight.GetDimensions(dims.x, dims.y);

	// oob is fine
	int2 lerpPxCoordA = int2(pxCoord - .5 * float2(isVertical, !isVertical));
	int2 lerpPxCoordB = int2(pxCoord + .5 * float2(isVertical, !isVertical));
	float heightA = TexHeight[lerpPxCoordA];
	float heightB = TexHeight[lerpPxCoordB];

	// normalize
	heightA = lerp(PosRange.x, PosRange.y, heightA);
	heightB = lerp(PosRange.x, PosRange.y, heightB);
	heightA = (heightA - ZRange.x) / (ZRange.y - ZRange.x);
	heightB = (heightB - ZRange.x) / (ZRange.y - ZRange.x);

	bool inBoundA = all(lerpPxCoordA > 0);
	bool inBoundB = all(lerpPxCoordB < int2(dims));
	if (inBoundA && inBoundB)
		return lerp(heightA, heightB, frac((isVertical ? pxCoord.x : pxCoord.y) - .5));
	else if (!inBoundA)
		return heightB;
	else
		return heightA;
}

// PR #2617 fixed visible stale/incremental terrain-shadow refreshes by allowing
// an immediate full-map refresh on transitions.  This custom AIO's DLL predates
// that C++ ABI and also contains Physical Sky / Procedural Grass, so dropping in
// the upstream DLL is unsafe.  The compatibility backport below performs the
// equivalent no-stale-map operation at the start of each existing update cycle:
// one dispatch walks every major-axis segment before publishing the new map.
// Subsequent 1.2.0 stripe dispatches are ignored until the next cycle start.

#define NTHREADS 128
groupshared float2 g_shadowHeight[NTHREADS];
groupshared float2 g_carryHeight;

uint GetWrappedCoord(int coord, uint dimension)
{
	uint magnitude = uint(abs(coord)) % dimension;
	return coord < 0 ? (dimension - magnitude) % dimension : magnitude;
}

[numthreads(NTHREADS, 1, 1)]
void main(const uint gtid : SV_GroupThreadID, const uint gid : SV_GroupID)
{
	uint2 dims;
	TexHeight.GetDimensions(dims.x, dims.y);

	bool isVertical = abs(LightPxDir.y) > abs(LightPxDir.x);
	uint majorDimension = isVertical ? dims.y : dims.x;
	uint minorDimension = isVertical ? dims.x : dims.y;
	float majorDirection = isVertical ? LightPxDir.y : LightPxDir.x;
	int majorStep = majorDirection > 0.0 ? 1 : -1;
	uint edgePxCoord = majorStep > 0 ? 0 : majorDimension - 1;

	// TerrainShadows 1.2.0 advances StartPxCoord by 128 each frame.  At index 0
	// the CPU also samples the current sun direction.  Do the complete refresh on
	// that frame and suppress the rolling partial writes on the remaining frames.
	// This preserves the original average update workload while removing the
	// visible band/chunk sweep across the world.
	if (StartPxCoord != edgePxCoord)
		return;

	float2 lightUVDir = LightPxDir * PxSize;
	uint2 rayStartPxCoord = isVertical ? uint2(gid, edgePxCoord) : uint2(edgePxCoord, gid);
	float2 rayStartUV = (rayStartPxCoord + .5) * PxSize;

	if (gtid == 0)
		g_carryHeight = float2(-1.0e20, -1.0e20);
	GroupMemoryBarrierWithGroupSync();

	uint segmentCount = (majorDimension + NTHREADS - 1) / NTHREADS;

	[loop]
	for (uint segment = 0; segment < segmentCount; ++segment)
	{
		uint segmentBase = segment * NTHREADS;
		uint globalStep = segmentBase + gtid;
		bool isValid = globalStep < majorDimension;

		float2 rawThreadUV = rayStartUV + globalStep * lightUVDir;
		float2 threadUV = rawThreadUV - floor(rawThreadUV);  // wraparound on minor axis
		float2 threadPxCoord = threadUV * dims;

		if (isValid)
		{
			float2 heights = GetInterpolatedHeight(threadPxCoord, isVertical).xx;

			// Carry the scan across 128-pixel segments in shared memory.  Do not
			// carry across a wrapped UV boundary, matching the original shader.
			if (gtid == 0 && segment > 0 && all(floor(rawThreadUV - lightUVDir) == floor(rawThreadUV)))
			{
				float2 sampleHeights = g_carryHeight + LightDeltaZ;
				heights = heights.x > sampleHeights.x ? heights : sampleHeights;
			}

			g_shadowHeight[gtid] = heights;
		}
		else
		{
			g_shadowHeight[gtid] = float2(-1.0e20, -1.0e20);
		}

		GroupMemoryBarrierWithGroupSync();

		// Parallel prefix scan for this 128-pixel segment.  Values are staged
		// around each barrier so no thread reads a value another thread has
		// already overwritten during the same scan step (also fixed upstream).
		[unroll]
		for (uint offset = 1; offset < NTHREADS; offset <<= 1)
		{
			bool combineHeights = false;
			float2 currentHeights = 0.0;
			float2 sampleHeights = 0.0;

			if (isValid && gtid >= offset)
			{
				if (all(floor(rawThreadUV - lightUVDir * offset) == floor(rawThreadUV)))
				{
					combineHeights = true;
					currentHeights = g_shadowHeight[gtid];
					sampleHeights = g_shadowHeight[gtid - offset] + LightDeltaZ * offset;
				}
			}

			GroupMemoryBarrierWithGroupSync();

			if (combineHeights)
				g_shadowHeight[gtid] = currentHeights.x > sampleHeights.x ? currentHeights : sampleHeights;

			GroupMemoryBarrierWithGroupSync();
		}

		if (isValid)
		{
			uint majorPxCoord = majorStep > 0 ? globalStep : (majorDimension - 1 - globalStep);
			float minorOffset = 0.5 + globalStep * (isVertical ? LightPxDir.x : LightPxDir.y);
			uint minorPxCoord = GetWrappedCoord(int(gid) + int(floor(minorOffset)), minorDimension);
			uint2 outputPxCoord = isVertical ? uint2(minorPxCoord, majorPxCoord) : uint2(majorPxCoord, minorPxCoord);

			// Immediate refresh: replace stale data instead of blending 50% of it
			// back in.  This is the behavior PR #2617 requests for transitions.
			RWTexShadowHeights[outputPxCoord] = g_shadowHeight[gtid];
		}

		GroupMemoryBarrierWithGroupSync();

		uint validCount = min(NTHREADS, majorDimension - segmentBase);
		if (gtid == 0)
			g_carryHeight = g_shadowHeight[validCount - 1];

		GroupMemoryBarrierWithGroupSync();
	}
}
