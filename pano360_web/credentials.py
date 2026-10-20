"""환경변수와 로컬 파일에서 API 키를 읽는다."""

import os


def read_env_file(path):
    """KEY=VALUE 형식(.env) 파일 읽기. 따옴표·export·주석 허용."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            out[k] = v
    return out


def read_key_file(path, var=None):
    """키 파일 읽기. 확장자 없는 경로면 .txt/.env/.key 도 시도.
    내용이 KEY=VALUE 형식이면 var 값을, 아니면 파일 내용 전체(첫 줄)를 키로 사용."""
    if not path:
        return ""
    for p in (path, path + ".txt", path + ".env", path + ".key", path + ".py"):
        if os.path.isfile(p):
            kv = read_env_file(p)
            if kv:
                if var and var in kv:
                    return kv[var]
                if len(kv) == 1:
                    return next(iter(kv.values()))
            with open(p, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip().strip('"').strip("'")
                    if line and not line.startswith("#") and "=" not in line:
                        return line
    return ""


def resolve_keys(cfg, base_dir):
    """config.json 설정대로 키를 찾아 {"openai": ..., "comfy_cloud": ...} 반환.
    우선순위: 환경변수 > env_files > 키 파일(openai_key_file 등) > config 에 직접 쓴 값"""
    env = {}
    for p in reversed(cfg.get("env_files", [])):
        p = p if os.path.isabs(p) else os.path.join(base_dir, p)
        env.update(read_env_file(p))
    env.update(os.environ)
    ok_var = cfg.get("openai_key_var", "OPENAI_API_KEY")
    cc_var = cfg.get("comfy_cloud_key_var", "COMFY_CLOUD_API_KEY")
    return {
        "openai": env.get(ok_var) or read_key_file(cfg.get("openai_key_file"), ok_var) or cfg.get("openai_api_key", ""),
        "comfy_cloud": env.get(cc_var) or read_key_file(cfg.get("comfy_cloud_key_file"), cc_var)
                       or cfg.get("comfy_cloud_api_key", ""),
    }
