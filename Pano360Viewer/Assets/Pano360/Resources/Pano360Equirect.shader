// 등장방형(Equirectangular) 360 사진을 구 안쪽에 그리는 언릿 셰이더.
// UV를 픽셀마다 시선 방향에서 계산하므로 구 메시 해상도와 무관하게 정확하고 극점 왜곡/이음새가 없습니다.
// _Stereo = 1 이면 위/아래(Top-Bottom) 스테레오 이미지로 보고 위쪽 절반을 왼쪽 눈에 사용합니다.
Shader "Pano360/Equirect"
{
    Properties
    {
        _MainTex ("Equirect Texture", 2D) = "black" {}
        _Stereo ("Stereo Top/Bottom", Float) = 0
        _Fade ("Fade To Black", Range(0, 1)) = 0
        _BlendTex ("Event Video Texture", 2D) = "black" {}
        _Blend ("Blend To Event Video", Range(0, 1)) = 0
        _BlendMask ("Event Video Mask (white = video)", 2D) = "white" {}
        _UseBlendMask ("Use Event Video Mask", Float) = 0
        _MaskLod ("Mask Softness (mip level)", Float) = 3
        _Wipe ("Mask Wipe (0 = all video, 1 = all photo)", Range(0, 1)) = 0
        _WipeField ("Wipe Distance Field (0 = mask edge, 1 = deepest)", 2D) = "white" {}
        _WipeSoft ("Wipe Edge Softness", Range(0.01, 0.5)) = 0.08
        _WipeNoise ("Wipe Irregularity", Range(0, 0.6)) = 0.25
    }
    SubShader
    {
        Tags { "Queue"="Background" "RenderType"="Background" "IgnoreProjector"="True" "PreviewType"="Skybox" }
        Cull Front
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma target 3.0
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _BlendTex;   // 사건 영상 (원본 사진 → 영상 → 원본 사진 전환용)
            float _Stereo;
            float _Fade;
            float _Blend;
            sampler2D _BlendMask;  // 사건 영상 범위 (흰색만 영상, 나머지는 고화질 사진 그대로)
            float _UseBlendMask;
            float _MaskLod;        // 마스크 밉맵 단계로 경계를 부드럽게 (3 ≈ 8배 축소 = 약 1.4°)
            float _Wipe;           // 끝날 때 마스크 범위를 가장자리부터 안쪽으로 지워 사진으로 (0 → 1)
            sampler2D _WipeField;  // 마스크 가장자리에서 안쪽까지의 거리 (0~1) — 이 순서로 벗겨짐
            float _WipeSoft;
            float _WipeNoise;

            float hash21(float2 p)
            {
                p = frac(p * float2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return frac(p.x * p.y);
            }
            float vnoise(float2 p)      // 값 노이즈 (지워지는 경계를 물결처럼 불규칙하게)
            {
                float2 i = floor(p), f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash21(i), b = hash21(i + float2(1, 0)), c = hash21(i + float2(0, 1)), d = hash21(i + float2(1, 1));
                return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
            }

            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 dir : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.dir = v.vertex.xyz; // 오브젝트 공간 방향 (구의 회전 = 사진 Yaw)
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float3 d = normalize(i.dir);
                float u = atan2(d.x, d.z) * (0.5 / UNITY_PI) + 0.5;
                float v = asin(clamp(d.y, -1.0, 1.0)) / UNITY_PI + 0.5;

                float2 monoUV = float2(u, v);   // 마스크는 모노 360 좌표
                if (_Stereo > 0.5)
                {
                #if defined(USING_STEREO_MATRICES)
                    float eye = unity_StereoEyeIndex;
                #else
                    float eye = 0;
                #endif
                    v = v * 0.5 + (eye < 0.5 ? 0.5 : 0.0);
                }

                // 밉맵이 없으므로 LOD 0 고정 샘플링 (u 이음새에서의 미분 튐 방지)
                fixed4 c = tex2Dlod(_MainTex, float4(u, v, 0, 0));
                if (_Blend > 0.001)
                {
                    float m = _UseBlendMask > 0.5 ? tex2Dlod(_BlendMask, float4(monoUV, 0, _MaskLod)).r : 1.0;
                    if (_Wipe > 0.0001 && _UseBlendMask > 0.5)
                    {
                        // 가장자리 거리 → 문턱을 올리면 끝부분부터 안쪽으로 마스크가 벗겨지며 사진으로
                        float field = tex2Dlod(_WipeField, float4(monoUV, 0, 0)).r;
                        float n = vnoise(monoUV * float2(48.0, 24.0)) * 0.65 + vnoise(monoUV * float2(140.0, 70.0)) * 0.35;
                        field = saturate(field * (1.0 - _WipeNoise) + n * _WipeNoise);
                        float th = _Wipe * (1.0 + 2.0 * _WipeSoft) - _WipeSoft;
                        m *= smoothstep(th - _WipeSoft, th + _WipeSoft, field);
                    }
                    c = lerp(c, tex2Dlod(_BlendTex, float4(u, v, 0, 0)), _Blend * m);
                }
                c.rgb *= (1.0 - _Fade);
                c.a = 1;
                return c;
            }
            ENDCG
        }
    }
    Fallback Off
}
