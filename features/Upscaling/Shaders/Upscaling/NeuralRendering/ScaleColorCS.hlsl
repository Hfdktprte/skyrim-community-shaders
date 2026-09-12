Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Destination : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
	uint sourceWidth, sourceHeight;
	uint destinationWidth, destinationHeight;
	Source.GetDimensions(sourceWidth, sourceHeight);
	Destination.GetDimensions(destinationWidth, destinationHeight);
	if (dispatchThreadId.x >= destinationWidth || dispatchThreadId.y >= destinationHeight)
		return;

	float2 sourcePosition = (float2(dispatchThreadId.xy) + 0.5) *
		(float2(sourceWidth, sourceHeight) / float2(destinationWidth, destinationHeight)) - 0.5;
	int2 basePosition = int2(floor(sourcePosition));
	float2 blend = frac(sourcePosition);
	int2 maximum = int2(sourceWidth - 1, sourceHeight - 1);
	int2 p00 = clamp(basePosition, int2(0, 0), maximum);
	int2 p10 = clamp(basePosition + int2(1, 0), int2(0, 0), maximum);
	int2 p01 = clamp(basePosition + int2(0, 1), int2(0, 0), maximum);
	int2 p11 = clamp(basePosition + int2(1, 1), int2(0, 0), maximum);
	float4 top = lerp(Source.Load(int3(p00, 0)), Source.Load(int3(p10, 0)), blend.x);
	float4 bottom = lerp(Source.Load(int3(p01, 0)), Source.Load(int3(p11, 0)), blend.x);
	Destination[dispatchThreadId.xy] = lerp(top, bottom, blend.y);
}

