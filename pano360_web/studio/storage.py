"""작업 경로와 meta.json 저장을 한 곳에서 관리한다."""

import json
import os
from pathlib import Path
import tempfile


class InvalidRunPath(ValueError):
    pass


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id, *parts):
        if (not isinstance(run_id, str) or not run_id or
                any(c in run_id for c in ("/", "\\", "..", ":"))):
            raise InvalidRunPath("bad run id")
        run_dir = (self.root / run_id).resolve()
        path = run_dir.joinpath(*parts).resolve()
        if not run_dir.is_relative_to(self.root) or not path.is_relative_to(run_dir):
            raise InvalidRunPath("bad run path")
        return str(path)

    def read(self, run_id):
        with open(self.path(run_id, "meta.json"), encoding="utf-8") as fp:
            return json.load(fp)

    def write(self, run_id, meta):
        target = self.path(run_id, "meta.json")
        # 서로 다른 저장 요청이 같은 임시 파일을 덮어쓰지 않도록 한다.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=os.path.dirname(target), delete=False) as fp:
            temp_path = fp.name
            try:
                json.dump(meta, fp, ensure_ascii=False, indent=1)
            except BaseException:
                fp.close()
                os.unlink(temp_path)
                raise
        try:
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
