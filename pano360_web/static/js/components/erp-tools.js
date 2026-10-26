// 1단계 안: 이미 360(2:1) 이미지일 때 — 바로 결과로 가져오기 · 8K 화질 개선 · 발밑·하늘 보정
import {post} from "../core/api.js";
import {$, errHtml, fillBackends} from "../core/dom.js";
import {store} from "../core/store.js";
import {submitJob, refreshRun, setTarget} from "../core/actions.js";
import {requestButton} from "../core/request.js";

export function setupErpTools() {
  requestButton($("importbtn"), async () => {
    const id = store.runId();
    if (!id) return false;
    $("importstate").textContent = "등록 중…";
    try {
      const att = await post(`/api/run/${id}/import`, {});
      if (!store.isRun(id)) return;
      $("importstate").textContent = `#${att.n} 결과로 등록했어요 — 아래 5·6단계를 쓰세요.`;
      await refreshRun();
      const a = store.attempts.find(x => x.n === att.n);
      if (a) { setTarget(a); store.emit("show-result", a); }
      store.emit("runs:changed");
    } catch (e) { $("importstate").innerHTML = errHtml(e.message); throw e; }
  });

  requestButton($("enhance"), () =>
    submitJob({backend: $("backend2").value, mode: "enhance"}, () => "화질 개선 대기 중…"));

  requestButton($("polebtn2"), () =>
    submitJob({backend: $("backend2").value, mode: "pole", from_attempt: null, settings: store.services.settings?.()}, () => "발밑·하늘 보정 대기 중…"));

  store.on("cfg", cfg => cfg && fillBackends($("backend2"), cfg));
}
