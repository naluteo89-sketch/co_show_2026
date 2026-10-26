// 2단계: 분석 결과 · 배치(yaw/pitch/화각) · 컨트롤 이미지 미리보기
import {post} from "../core/api.js";
import {$, show, errHtml} from "../core/dom.js";
import {store} from "../core/store.js";

export function setupAnalysis() {
  let timer = null, request = 0;

  const placement = () => ({yaw: +$("yaw").value, pitch: +$("pitch").value, hfov: +$("hfov").value,
                            vfov: $("vfov").value === "" ? null : +$("vfov").value});
  store.services.placement = placement;

  function refreshControl() {
    clearTimeout(timer);
    const id = store.runId();
    if (!id) return;
    const mine = ++request, pl = placement();
    timer = setTimeout(async () => {
      try {
        const r = await post("/api/control", {run_id: id, placement: pl});
        if (store.isRun(id) && mine === request) { $("ctrlimg").src = r.url; $("ctrlstate").textContent = ""; }
      } catch (e) { if (mine === request) $("ctrlstate").innerHTML = errHtml(`미리보기 실패: ${e.message}`); }
    }, 300);
  }

  store.on("analysis", ({analysis: a, placement: pl}) => {
    show("anacard");
    $("summary").textContent = a.summary_ko || "";
    $("tips").textContent = "주의 · " + (a.tips_ko || "");
    $("yaw").value = pl.yaw; $("pitch").value = pl.pitch; $("hfov").value = pl.hfov; $("vfov").value = pl.vfov ?? "";
    $("hfovsrc").textContent = pl.hfov_source ? `· ${pl.hfov_source}` : "";
    refreshControl();
  });

  store.on("run", m => {
    clearTimeout(timer); request++;
    if (!m) { show("anacard", false); return; }
    if (m.analysis) {
      const last = (m.attempts || []).slice().reverse().find(a => a.placement?.hfov && a.prompts?.positive);
      store.emit("analysis", {analysis: m.analysis, placement: last ? last.placement : m.placement,
                              prompts: last ? last.prompts : m.prompts});
    } else show("anacard", false);
  });

  ["yaw", "pitch", "hfov", "vfov"].forEach(id => $(id).addEventListener("input", refreshControl));
}
