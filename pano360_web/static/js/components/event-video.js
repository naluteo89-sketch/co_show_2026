// 6단계 안: 영상 만들기 (노드 파일에 값만 넣어 실행) · 진행 확인 · 결과 받기
import {api, post, esc} from "../core/api.js";
import {$, errHtml, fillBackends} from "../core/dom.js";
import {store} from "../core/store.js";
import {requestButton} from "../core/request.js";

export function setupEventVideo() {
  let runId = null, n = null;
  // ---- 영상 만들기: 마스크 인페인팅 노드 (화질 올린 360 + 칠한 범위 + 프롬프트)
  let vTimer = null;
  function vidShow(v) {
    clearTimeout(vTimer);
    $("evout").classList.toggle("hidden", !(v && v.state === "done"));
    if (!v || v.state === "none") { $("evstate").textContent = ""; return; }
    const nodeLink = v.node_url ? ` · <a href="${v.node_url}" download="${runId}_video_node.json">값 넣은 노드 파일 받기</a>` : "";
    const maskTxt = v.mask_mode === "rect" && v.rect ? ` · 마스크 사각형 x${v.rect[0]} y${v.rect[1]} ${v.rect[2]}×${v.rect[3]}` : v.mask_mode === "image" ? " · 마스크 그림" : "";
    if (v.state === "running" || v.state === "composing") {   // composing = 서버가 4K 360 에 붙이는 중
      const pr = v.progress || {};
      $("evstate").innerHTML = esc(`영상 만드는 중… ${pr.state === "running" ? (pr.node_title || pr.node || "") + (pr.max ? ` ${pr.value}/${pr.max}` : "") : "대기 중"}`) + esc(maskTxt) + nodeLink;
      vTimer = setTimeout(vidPoll, 5000);
      return;
    }
    if (v.state === "error") { $("evstate").innerHTML = errHtml(v.error || "실패") + nodeLink; return; }
    $("evstate").innerHTML = esc(`완료 · ${v.finished || ""}${maskTxt}`) + nodeLink;
    const main = v.audio_url || v.silent_url;
    if (!$("evvideo").src.includes(main.split("?")[0])) $("evvideo").src = main;
    const base = `${runId}_a${String(n).padStart(2, "0")}_flood_4k`;
    $("evdl").classList.toggle("hidden", !v.audio_url); $("evdl").href = v.audio_url || "#"; $("evdl").setAttribute("download", base + "_audio.mp4");
    $("evdl2").classList.toggle("hidden", !v.silent_url); $("evdl2").href = v.silent_url || "#"; $("evdl2").setAttribute("download", base + ".mp4");
    $("evinfo").textContent = `${v.template || ""} · ${v.seconds}초 · ${v.fps || (v.out || {}).fps || 24}fps · seed ${v.seed}${v.lora_strength != null ? " · LoRA " + v.lora_strength : ""}`;
  }
  async function vidPoll() {
    if (!runId) return;
    const id = runId, k = n;
    try { const v = await api(`/api/run/${id}/attempt/${k}/event/video`); if (id === runId && k === n) vidShow(v); }
    catch (e) { if (id === runId && k === n) { $("evstate").innerHTML = errHtml(`상태 확인 실패: ${e.message} — 다시 확인해요`); vTimer = setTimeout(vidPoll, 8000); } }
  }
  requestButton($("evgo"), async () => {
    if (!runId) return false;
    const id = runId, k = n;
    $("evstate").textContent = "범위·이미지를 올리고 작업을 보내는 중…";
    try {
      await store.services.mask?.flush();
      const v = await post(`/api/run/${id}/attempt/${k}/event/video`, {
        backend: $("evbackend").value, seconds: +$("evsec").value, lora_strength: +$("evlora").value,
        positive: $("eltxpos").value, negative: $("eltxneg").value});
      if (id === runId && k === n) vidShow(v);
    } catch (e) { if (id === runId && k === n) $("evstate").innerHTML = errHtml(e.message); throw e; }
  });

  // ---- 평소 움직임 (반복용): 같은 범위, 사건 없이 평소 흐름만 → 뷰어에서 평소 → 사건 → 평소 로 이어 붙임
  let aTimer = null;
  function ambShow(v) {
    clearTimeout(aTimer);
    const vEl = $("evambvideo");
    if (!v || v.state === "none") { vEl.classList.add("hidden"); return; }
    if (v.state === "running" || v.state === "composing") {
      const pr = v.progress || {};
      $("evambstate").textContent = `평소 움직임 만드는 중… ${pr.state === "running" ? (pr.node_title || pr.node || "") + (pr.max ? ` ${pr.value}/${pr.max}` : "") : "대기 중"}`;
      aTimer = setTimeout(ambPoll, 5000);
      return;
    }
    if (v.state === "error") { $("evambstate").innerHTML = errHtml(v.error || "실패"); vEl.classList.add("hidden"); return; }
    $("evambstate").textContent = `평소 움직임 완료 · ${v.finished || ""} · ${v.seconds}초 — 'Quest 뷰어에 추가'를 누르면 평소 → 사건 → 평소 반복 영상으로 넣어요.`;
    const main = v.audio_url || v.silent_url;
    if (main && !vEl.src.includes(main.split("?")[0])) vEl.src = main;
    vEl.classList.remove("hidden");
  }
  async function ambPoll() {
    if (!runId) return;
    const id = runId, k = n;
    try { const v = await api(`/api/run/${id}/attempt/${k}/event/video?kind=ambient`); if (id === runId && k === n) ambShow(v); }
    catch (e) { if (id === runId && k === n) { $("evambstate").innerHTML = errHtml(`상태 확인 실패: ${e.message} — 다시 확인해요`); aTimer = setTimeout(ambPoll, 8000); } }
  }
  requestButton($("evamb"), async () => {
    if (!runId) return false;
    const id = runId, k = n;
    $("evambstate").textContent = "범위·이미지를 올리고 작업을 보내는 중…";
    try {
      await store.services.mask?.flush();
      const v = await post(`/api/run/${id}/attempt/${k}/event/video`, {
        kind: "ambient", backend: $("evbackend").value, seconds: +$("evsec").value, negative: $("eltxneg").value});
      if (id === runId && k === n) ambShow(v);
    } catch (e) { if (id === runId && k === n) $("evambstate").innerHTML = errHtml(e.message); throw e; }
  });

  store.on("cfg", cfg => cfg && fillBackends($("evbackend"), cfg));
  store.on("target", att => {
    clearTimeout(vTimer); clearTimeout(aTimer);
    if (!att) { runId = n = null; return; }
    runId = store.runId(); n = att.n;
    const v = att.event && att.event.video;
    if (v && v.seconds) $("evsec").value = String(v.seconds);
    if (v && v.lora_strength != null) $("evlora").value = v.lora_strength;
    vidShow(v);
    ambShow(att.event && att.event.ambient);
  });
}
