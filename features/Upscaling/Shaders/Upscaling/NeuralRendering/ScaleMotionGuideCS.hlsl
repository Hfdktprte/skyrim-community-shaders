Texture2D<float2> Source : register(t0);
RWTexture2D<float2> Destination : register(u0);

cbuffer ScaleMotionGuideCB : register(b0)
{
	uint2 SourceActiveDim;
	uint2 DestinationDim;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
	if (any(dispatchThreadId.xy >= DestinationDim))
		return;

	float2 sourcePosition = (float2(dispatchThreadId.xy) + 0.5f) *
		(float2(SourceActiveDim) / float2(DestinationDim)) - 0.5f;
	int2 basePosition = int2(floor(sourcePosition));
	float2 blend = frac(sourcePosition);
	int2 maximum = int2(SourceActiveDim) - 1;
	int2 p00 = clamp(basePosition, int2(0, 0), maximum);
	int2 p10 = clamp(basePosition + int2(1, 0), int2(0, 0), maximum);
	int2 p01 = clamp(basePosition + int2(0, 1), int2(0, 0), maximum);
	int2 p11 = clamp(basePosition + int2(1, 1), int2(0, 0), maximum);
	float2 top = lerp(Source.Load(int3(p00, 0)), Source.Load(int3(p10, 0)), blend.x);
	float2 bottom = lerp(Source.Load(int3(p01, 0)), Source.Load(int3(p11, 0)), blend.x);
	Destination[dispatchThreadId.xy] = lerp(top, bottom, blend.y);
}
