// FidelityFX Single Pass Downsampler (SPD)
// Based on https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/upscalers/fsr3/include/gpu/spd/ffx_spd.h
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// D3D11 exposes eight UAV slots. Grass only needs six output mips.
#define SPD_MAX_MIPS 6

globallycoherent RWTexture2D<float> SpdMip1 : register(u0);
globallycoherent RWTexture2D<float> SpdMip2 : register(u1);
globallycoherent RWTexture2D<float> SpdMip3 : register(u2);
globallycoherent RWTexture2D<float> SpdMip4 : register(u3);
globallycoherent RWTexture2D<float> SpdMip5 : register(u4);
globallycoherent RWTexture2D<float> SpdMip6 : register(u5);
Texture2D<float> SpdSource : register(t0);

cbuffer SpdParams : register(b0)
{
    uint2 SpdSourceSize; // dimensions of the source mip
    uint SpdMipCount; // output mips to produce, 1..12
    uint SpdNumWorkGroups; // groups dispatched, so the last one can finish the tail
};

groupshared float spdIntermediate[16][16];
float SpdLoadSource(int2 tex)
{
    return SpdSource.Load(int3(clamp(tex, int2(0, 0), int2(SpdSourceSize) - 1), 0));
}

void SpdStore(int2 pix, float value, uint index)
{
    switch (index)
    {
        case 0:
            SpdMip1[pix] = value;
            break;
        case 1:
            SpdMip2[pix] = value;
            break;
        case 2:
            SpdMip3[pix] = value;
            break;
        case 3:
            SpdMip4[pix] = value;
            break;
        case 4:
            SpdMip5[pix] = value;
            break;
        case 5:
            SpdMip6[pix] = value;
            break;
        default:
            break;
    }
}

float SpdReduce4(float a, float b, float c, float d)
{
    return max(max(a, b), max(c, d));
}

void SpdStoreIntermediate(uint x, uint y, float value)
{
    spdIntermediate[x][y] = value;
}

float SpdLoadIntermediate(uint x, uint y)
{
    return spdIntermediate[x][y];
}

float SpdReduceIntermediate(uint2 i0, uint2 i1, uint2 i2, uint2 i3)
{
    float v0 = SpdLoadIntermediate(i0.x, i0.y);
    float v1 = SpdLoadIntermediate(i1.x, i1.y);
    float v2 = SpdLoadIntermediate(i2.x, i2.y);
    float v3 = SpdLoadIntermediate(i3.x, i3.y);
    return SpdReduce4(v0, v1, v2, v3);
}

float SpdReduceLoadSource(int2 base)
{
    float v0 = SpdLoadSource(base + int2(0, 0));
    float v1 = SpdLoadSource(base + int2(0, 1));
    float v2 = SpdLoadSource(base + int2(1, 0));
    float v3 = SpdLoadSource(base + int2(1, 1));
    return SpdReduce4(v0, v1, v2, v3);
}

void SpdDownsampleMips_0_1(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip)
{
    float v[4];

    int2 tex = int2(workGroupID.xy * 64) + int2(x * 2, y * 2);
    int2 pix = int2(workGroupID.xy * 32) + int2(x, y);
    v[0]     = SpdReduceLoadSource(tex);
    SpdStore(pix, v[0], 0);

    tex  = int2(workGroupID.xy * 64) + int2(x * 2 + 32, y * 2);
    pix  = int2(workGroupID.xy * 32) + int2(x + 16, y);
    v[1] = SpdReduceLoadSource(tex);
    SpdStore(pix, v[1], 0);

    tex  = int2(workGroupID.xy * 64) + int2(x * 2, y * 2 + 32);
    pix  = int2(workGroupID.xy * 32) + int2(x, y + 16);
    v[2] = SpdReduceLoadSource(tex);
    SpdStore(pix, v[2], 0);

    tex  = int2(workGroupID.xy * 64) + int2(x * 2 + 32, y * 2 + 32);
    pix  = int2(workGroupID.xy * 32) + int2(x + 16, y + 16);
    v[3] = SpdReduceLoadSource(tex);
    SpdStore(pix, v[3], 0);

    if (mip <= 1)
        return;

    for (uint i = 0; i < 4; i++)
    {
        SpdStoreIntermediate(x, y, v[i]);
        GroupMemoryBarrierWithGroupSync();
        if (localInvocationIndex < 64)
        {
            v[i] = SpdReduceIntermediate(uint2(x * 2 + 0, y * 2 + 0), uint2(x * 2 + 1, y * 2 + 0), uint2(x * 2 + 0, y * 2 + 1), uint2(x * 2 + 1, y * 2 + 1));
            SpdStore(int2(workGroupID.xy * 16) + int2(x + (i % 2) * 8, y + (i / 2) * 8), v[i], 1);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localInvocationIndex < 64)
    {
        SpdStoreIntermediate(x + 0, y + 0, v[0]);
        SpdStoreIntermediate(x + 8, y + 0, v[1]);
        SpdStoreIntermediate(x + 0, y + 8, v[2]);
        SpdStoreIntermediate(x + 8, y + 8, v[3]);
    }
}

void SpdDownsampleMip_2(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip)
{
 if (localInvocationIndex < 64)
    {
        float v = SpdReduceIntermediate(uint2(x * 2 + 0, y * 2 + 0), uint2(x * 2 + 1, y * 2 + 0), uint2(x * 2 + 0, y * 2 + 1), uint2(x * 2 + 1, y * 2 + 1));
        SpdStore(int2(workGroupID.xy * 8) + int2(x, y), v, mip);
        // store to LDS, try to reduce bank conflicts
        SpdStoreIntermediate(x * 2 + y % 2, y * 2, v);
    }
}

void SpdDownsampleMip_3(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip)
{
    if (localInvocationIndex < 16)
    {
        float v = SpdReduceIntermediate(uint2(x * 4 + 0 + 0, y * 4 + 0), uint2(x * 4 + 2 + 0, y * 4 + 0), uint2(x * 4 + 0 + 1, y * 4 + 2), uint2(x * 4 + 2 + 1, y * 4 + 2));
        SpdStore(int2(workGroupID.xy * 4) + int2(x, y), v, mip);
        // store to LDS

        SpdStoreIntermediate(x * 4 + y, y * 4, v);
    }
}

void SpdDownsampleMip_4(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint mip)
{
    if (localInvocationIndex < 4)
    {
        float v = SpdReduceIntermediate(uint2(x * 8 + 0 + 0 + y * 2, y * 8 + 0),
                                         uint2(x * 8 + 4 + 0 + y * 2, y * 8 + 0),
                                         uint2(x * 8 + 0 + 1 + y * 2, y * 8 + 4),
                                         uint2(x * 8 + 4 + 1 + y * 2, y * 8 + 4));
        SpdStore(int2(workGroupID.xy * 2) + int2(x, y), v, mip);
        SpdStoreIntermediate(x + y * 2, 0, v);
    }
}

void SpdDownsampleMip_5(uint2 workGroupID, uint localInvocationIndex, uint mip)
{
    if (localInvocationIndex < 1)
    {
        float v = SpdReduceIntermediate(uint2(0, 0), uint2(1, 0), uint2(2, 0), uint2(3, 0));
        SpdStore(int2(workGroupID.xy), v, mip);
    }
}

void SpdDownsampleNextFour(uint x, uint y, uint2 workGroupID, uint localInvocationIndex, uint baseMip, uint mips)
{
    if (mips <= baseMip)
        return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_2(x, y, workGroupID, localInvocationIndex, baseMip);

    if (mips <= baseMip + 1)
        return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_3(x, y, workGroupID, localInvocationIndex, baseMip + 1);

    if (mips <= baseMip + 2)
        return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_4(x, y, workGroupID, localInvocationIndex, baseMip + 2);

    if (mips <= baseMip + 3)
        return;
    GroupMemoryBarrierWithGroupSync();
    SpdDownsampleMip_5(workGroupID, localInvocationIndex, baseMip + 3);
}

uint ffxBitfieldExtract(uint src, uint off, uint bits)
{
    uint mask = (1u << bits) - 1;
    return (src >> off) & mask;
}

uint ffxBitfieldInsertMask(uint src, uint ins, uint bits)
{
    uint mask = (1u << bits) - 1;
    return (ins & mask) | (src & (~mask));
}

uint2 ffxRemapForWaveReduction(uint a)
{
    return uint2(ffxBitfieldInsertMask(ffxBitfieldExtract(a, 2u, 3u), a, 1u), ffxBitfieldInsertMask(ffxBitfieldExtract(a, 3u, 3u), ffxBitfieldExtract(a, 1u, 2u), 2u));
}

// Reduces one 64x64 tile to six levels, then the last group to arrive carries the remainder alone.
void SpdDownsample(uint2 workGroupID, uint localInvocationIndex, uint mips, uint numWorkGroups)
{
    // compute MIP level 0 and 1
    uint2        sub_xy = ffxRemapForWaveReduction(localInvocationIndex % 64);
    uint x      = sub_xy.x + 8 * ((localInvocationIndex >> 6) % 2);
    uint y      = sub_xy.y + 8 * ((localInvocationIndex >> 7));
    SpdDownsampleMips_0_1(x, y, workGroupID, localInvocationIndex, mips);

    // compute MIP level 2, 3, 4, 5
    SpdDownsampleNextFour(x, y, workGroupID, localInvocationIndex, 2, mips);

}

[numthreads(256, 1, 1)] void main(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
{
    SpdDownsample(WorkGroupId.xy, LocalThreadIndex, SpdMipCount, SpdNumWorkGroups);
}
