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
                    c = lerp(c, tex2Dlod(_BlendTex, float4(u, v, 0, 0)), _Blend);
                c.rgb *= (1.0 - _Fade);
                c.a = 1;
                return c;
            }
            ENDCG
        }
    }
    Fallback Off
}
