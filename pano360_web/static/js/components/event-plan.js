// 6단계: 사건 연출 (LLM 으로 만들기 · 직접 고쳐 '수정 반영')
// '사건 만들기'를 먼저 하지 않아도 칸을 채우고 '수정 반영'으로 바로 만들 수 있다.
import {post, esc} from "../core/api.js";
import {$, errHtml, show, copyText} from "../core/dom.js";
import {store} from "../core/store.js";
import {requestButton} from "../core/request.js";

const KEYS = ["yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end", "keep"];
const NUM = new Set(["yaw", "pitch", "hfov"]);

export function setupEventPlan() {
  let runId = null, n = null, ev = null, edited = false;
  const same = (id, k) => id === runId && k === n;

  function render() {
    const has = !!(ev && ev.view);
    show("eviewbox", has); show("eviewnone", !has); show("esummary", !!ev);
    if (ev) $("esummary").innerHTML = `<b>${esc(ev.name_ko || "")}</b> · ${esc(ev.summary_ko || "")}` +
      (ev.risk_ko ? `<br><span class="muted">주의 · ${esc(ev.risk_ko)}</span>` : "");
    if (has) {
      $("eview").src = ev.view; $("eviewa").href = ev.view;
      $("eviewdl").href = ev.view; $("eviewdl").setAttribute("download", ev.view_name || "event_view.png");
      $("eoverlay").src = ev.overlay || "";
    }
    KEYS.forEach(k => { $("e_" + k).value = ev?.[k] ?? ""; });
    $("e_pos").value = ev?.positive || ""; $("e_neg").value = ev?.negative || "";
    $("emeta").textContent = ev ? `${ev.model || ""} · ${ev.updated || ev.created || ""} · ${ev.seconds || 5}초 · 사건 화면 ${(ev.view_size || []).join("×")}` : "";
    edited = false;
  }

  function collect() {
    const out = {};
    KEYS.forEach(k => {
      const v = $("e_" + k).value;
      if (NUM.has(k)) { if (v !== "") out[k] = Number(v); } else out[k] = v;
    });
    return out;
  }

  KEYS.forEach(k => $("e_" + k).addEventListener("input", () => { edited = false; }));
  ["e_pos", "e_neg"].forEach(id => $(id).addEventListener("input", () => { edited = true; }));

  function updated(r) { ev = r; render(); store.emit("event", ev); }

  requestButton($("eplan"), async () => {
    if (!runId) return false;
    const id = runId, k = n;
    $("estate").textContent = "LLM 이 결과 360 을 보고 연출을 짜는 중… (20~60초)";
    try {
      const model = $("mmodel") ? $("mmodel").value : undefined;
      const r = await post(`/api/run/${id}/attempt/${k}/event`, {note: $("enote").value, seconds: +$("esec").value, model});
      if (!same(id, k)) return;
      $("estate").textContent = `${r.model} · 사건 화면과 프롬프트를 만들었어요.`;
      updated(r);
    } catch (e) { if (same(id, k)) $("estate").innerHTML = errHtml(e.message); throw e; }
  });

  requestButton($("esave"), async () => {
    if (!runId) return false;
    const id = runId, k = n;
    $("estate").textContent = "사건 화면과 프롬프트를 만드는 중…";
    try {
      const body = {...collect(), seconds: +$("esec").value, prompt_edited: edited};
      if (edited) { body.positive = $("e_pos").value; body.negative = $("e_neg").value; }
      const r = await post(`/api/run/${id}/attempt/${k}/event/update`, body);
      if (!same(id, k)) return;
      $("estate").textContent = "저장했어요.";
      updated(r);
    } catch (e) { if (same(id, k)) $("estate").innerHTML = errHtml(e.message); throw e; }
  });

  $("ecp").onclick = () => copyText($("e_pos").value, $("estate"), "Positive 를 복사했어요.");

  store.on("target", att => {
    if (!att) { show("eventcard", false); runId = n = ev = null; return; }
    runId = store.runId(); n = att.n; ev = att.event || null;
    show("eventcard");
    if (ev && ev.seconds) $("esec").value = String(Math.round(ev.seconds));
    $("estate").textContent = ev ? `#${att.n} 결과의 저장된 사건을 불러왔어요.` : "";
    render();
  });
}
