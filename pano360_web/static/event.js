// ---------- 6. 사건 영상: 한 번 일어나는 연출 (잔잔 → 물이 넘쳐 화면을 덮음 → 다시 잔잔)
// LLM 이 사건 위치·내용·프롬프트를 정하고, 사건 화면(16:9) + ComfyUI 워크플로를 준다.
// ComfyUI 에서 만든 MP4 를 올리면 원본에서 달라진 곳만 360 위에 섞어 MP4 로.

const KEYS = ["yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end", "keep"];
const NUM = new Set(["yaw", "pitch", "hfov"]);

export function setupEvent({api, post, esc}) {
  const $ = id => document.getElementById(id);
  let runId = null, n = null, ev = null, edited = false;

  function render() {
    $("ebody").classList.toggle("hidden", !ev);
    if (!ev) return;
    $("esummary").innerHTML = `<b>${esc(ev.name_ko || "")}</b> · ${esc(ev.summary_ko || "")}` +
      (ev.risk_ko ? `<br><span class="muted">주의 · ${esc(ev.risk_ko)}</span>` : "");
    $("eview").src = ev.view || ""; $("eviewa").href = ev.view || "";
    $("eviewdl").href = ev.view || ""; $("eviewdl").setAttribute("download", ev.view_name || "event_view.png");
    $("eoverlay").src = ev.overlay || "";
    KEYS.forEach(k => { $("e_" + k).value = ev[k] ?? ""; });
    $("e_pos").value = ev.positive || ""; $("e_neg").value = ev.negative || "";
    const wf = ev.workflows || {};
    $("ewf14").href = wf.wan14 || "#"; $("ewf14").setAttribute("download", `${runId}_event_wan14.json`);
    $("ewf5").href = wf.wan5 || "#"; $("ewf5").setAttribute("download", `${runId}_event_wan5.json`);
    $("emeta").textContent = `${ev.model || ""} · ${ev.updated || ev.created || ""} · ${ev.seconds || 5}초 · 사건 화면 ${(ev.view_size || []).join("×")}`;
    edited = false;
    showResult(ev.result);
  }

  function showResult(r) {
    $("eout").classList.toggle("hidden", !r);
    if (!r) return;
    if (!$("evideo").src.includes(r.video)) { $("evideo").poster = r.preview || ""; $("evideo").src = r.video; }
    $("edl").href = r.video;
    $("einfo").innerHTML = esc(`${(r.size || []).join("×")} · ${r.fps}fps · ${r.seconds}초 · 가장 많이 덮인 때 ${Math.round((r.peak_cover || 0) * 100)}% ` +
      `(시작 ${Math.round((r.start_cover || 0) * 100)}% · 끝 ${Math.round((r.end_cover || 0) * 100)}%) · 흔들림 보정 최대 ${r.max_shift_px}px`) +
      (r.alpha ? ` · <a href="${r.alpha}" target="_blank">섞인 범위 보기</a>` : "") +
      (r.preview ? ` · <a href="${r.preview}" target="_blank">절정 장면</a>` : "");
  }

  function collect() {
    const out = {};
    KEYS.forEach(k => { out[k] = NUM.has(k) ? Number($("e_" + k).value) : $("e_" + k).value; });
    return out;
  }

  KEYS.forEach(k => $("e_" + k).addEventListener("input", () => { edited = false; }));
  ["e_pos", "e_neg"].forEach(id => $(id).addEventListener("input", () => { edited = true; }));

  $("eplan").onclick = async () => {
    $("eplan").disabled = true; $("estate").textContent = "LLM 이 결과 360 을 보고 연출을 짜는 중… (20~60초)";
    try {
      const model = $("mmodel") ? $("mmodel").value : undefined;
      ev = await post(`/api/run/${runId}/attempt/${n}/event`, {note: $("enote").value, seconds: +$("esec").value, model});
      $("estate").textContent = `${ev.model} · 사건 화면과 워크플로를 만들었어요.`;
      render();
    } catch (e) { $("estate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("eplan").disabled = false;
  };

  $("esave").onclick = async () => {
    if (!ev) return;
    $("esave").disabled = true; $("estate").textContent = "사건 화면과 워크플로를 다시 만드는 중…";
    try {
      const body = {...collect(), seconds: +$("esec").value, prompt_edited: edited};
      if (edited) { body.positive = $("e_pos").value; body.negative = $("e_neg").value; }
      ev = await post(`/api/run/${runId}/attempt/${n}/event/update`, body);
      $("estate").textContent = "저장했어요.";
      render();
    } catch (e) { $("estate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("esave").disabled = false;
  };

  $("ecp").onclick = async () => {
    try { await navigator.clipboard.writeText($("e_pos").value); $("estate").textContent = "Positive 를 복사했어요."; }
    catch { $("estate").textContent = "복사하지 못했어요. 칸에서 직접 복사하세요."; }
  };

  $("ecompose").onclick = async () => {
    const f = $("efile").files[0];
    if (!f) { $("ecstate").textContent = "ComfyUI 에서 만든 MP4 를 골라 주세요."; return; }
    $("ecompose").disabled = true; $("ecstate").textContent = "영상을 올리고 360 에 합치는 중… (4K 5초 기준 1~2분)";
    try {
      const fd = new FormData(); fd.append("video", f);
      const r = await api(`/api/run/${runId}/attempt/${n}/event/compose`, {method: "POST", body: fd});
      ev = {...ev, result: r};
      $("ecstate").textContent = `완료 · ${r.finished || ""}`;
      showResult(r);
    } catch (e) { $("ecstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("ecompose").disabled = false;
  };

  $("equest").onclick = async () => {
    try { const r = await post(`/api/run/${runId}/attempt/${n}/event/to_viewer`, {});
          $("einfo").textContent = `Quest 뷰어에 추가됨 (목록 ${r.count}개) — Unity 에서 다시 빌드하세요`; }
    catch (e) { $("einfo").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };

  return {
    show(id, att) {
      if (!att || att.state !== "done" || !(att.files || []).length) { this.hide(); return; }
      runId = id; n = att.n; ev = att.event || null;
      $("eventcard").classList.remove("hidden");
      $("etarget").textContent = `#${att.n} 결과로 만들어요`;
      if (ev && ev.seconds) $("esec").value = String(Math.round(ev.seconds));
      $("estate").textContent = ev ? "저장된 사건을 불러왔어요." : "";
      $("ecstate").textContent = "";
      render();
    },
    hide() { $("eventcard").classList.add("hidden"); ev = null; runId = null; },
  };
}
