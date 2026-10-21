// ---------- 6. 사건 영상: 한 번 일어나는 연출 (잔잔 → 물이 넘쳐 화면을 덮음 → 다시 잔잔)
// LLM 이 사건 위치·내용·프롬프트를 정하고 사건 화면(16:9)을 준다. 360 위에 움직일 범위(마스크)를 칠해 PNG 로.
// 영상 생성·합성은 ComfyUI(LTX) 쪽에서 한다.

const KEYS = ["yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end", "keep"];
const NUM = new Set(["yaw", "pitch", "hfov"]);

export function setupEvent({api, post, esc}) {
  const $ = id => document.getElementById(id);
  let runId = null, n = null, ev = null, edited = false;

  function render() {
    $("ebody").classList.toggle("hidden", !(ev && ev.view));
    if (!ev || !ev.view) return;
    $("esummary").innerHTML = `<b>${esc(ev.name_ko || "")}</b> · ${esc(ev.summary_ko || "")}` +
      (ev.risk_ko ? `<br><span class="muted">주의 · ${esc(ev.risk_ko)}</span>` : "");
    $("eview").src = ev.view || ""; $("eviewa").href = ev.view || "";
    $("eviewdl").href = ev.view || ""; $("eviewdl").setAttribute("download", ev.view_name || "event_view.png");
    $("eoverlay").src = ev.overlay || "";
    KEYS.forEach(k => { $("e_" + k).value = ev[k] ?? ""; });
    $("e_pos").value = ev.positive || ""; $("e_neg").value = ev.negative || "";
    $("emeta").textContent = `${ev.model || ""} · ${ev.updated || ev.created || ""} · ${ev.seconds || 5}초 · 사건 화면 ${(ev.view_size || []).join("×")}`;
    edited = false;
    if (ev.ltx360) { $("eltxpos").value = ev.ltx360.positive || ""; $("eltxneg").value = ev.ltx360.negative || ""; }
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
      $("estate").textContent = `${ev.model} · 사건 화면과 프롬프트를 만들었어요.`;
      render();
    } catch (e) { $("estate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("eplan").disabled = false;
  };

  $("esave").onclick = async () => {
    if (!ev) return;
    $("esave").disabled = true; $("estate").textContent = "사건 화면과 프롬프트를 다시 만드는 중…";
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

  // ---- 움직일 범위 칠하기 (360 그림 위, 흰색 = 영상)
  const cv = $("emask"), cx = cv.getContext("2d");
  let erasing = false, painting = false, maskDirty = false, last = null;
  function paintAt(e) {
    const b = cv.getBoundingClientRect();
    const x = (e.clientX - b.left) / b.width * cv.width, y = (e.clientY - b.top) / b.height * cv.height;
    const r = +$("ebrush").value / 2;
    cx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
    cx.strokeStyle = cx.fillStyle = "rgb(255,40,70)"; cx.lineWidth = r * 2; cx.lineCap = "round";
    if (last) { cx.beginPath(); cx.moveTo(last[0], last[1]); cx.lineTo(x, y); cx.stroke();
      // 360 좌우가 이어지므로 가장자리에서 칠하면 반대쪽에도
      if (x < r || x > cv.width - r) { const sh = x < r ? cv.width : -cv.width; cx.beginPath(); cx.moveTo(last[0] + sh, last[1]); cx.lineTo(x + sh, y); cx.stroke(); } }
    else { cx.beginPath(); cx.arc(x, y, r, 0, Math.PI * 2); cx.fill(); }
    last = [x, y]; maskDirty = true; $("emstate").textContent = "칠하는 중 — '범위 저장'을 눌러 주세요";
  }
  $("epaint").addEventListener("pointerdown", e => { if (e.button !== 0) return; e.preventDefault();
    $("epaint").setPointerCapture(e.pointerId); painting = true; last = null; paintAt(e); });
  $("epaint").addEventListener("pointermove", e => { if (painting) paintAt(e); });
  ["pointerup", "pointercancel"].forEach(t => $("epaint").addEventListener(t, () => { painting = false; last = null; }));
  $("eerase").onclick = () => { erasing = !erasing; $("eerase").textContent = erasing ? "칠하기" : "지우개"; };
  $("eclear").onclick = () => { cx.clearRect(0, 0, cv.width, cv.height); maskDirty = true; $("emstate").textContent = "지웠어요 — '범위 저장'을 눌러 주세요"; };
  function maskPng() {                                       // 칠한 곳(빨강) → 흰색 PNG
    const d = cx.getImageData(0, 0, cv.width, cv.height), t = document.createElement("canvas");
    t.width = cv.width; t.height = cv.height;
    const tx = t.getContext("2d"), o = tx.createImageData(cv.width, cv.height);
    let any = 0;
    for (let i = 0; i < d.data.length; i += 4) { const a = d.data[i + 3]; any += a > 0;
      o.data[i] = o.data[i + 1] = o.data[i + 2] = a; o.data[i + 3] = 255; }
    tx.putImageData(o, 0, 0);
    return any ? t.toDataURL("image/png") : "";
  }
  async function saveMask() {
    const r = await post(`/api/run/${runId}/attempt/${n}/event/mask`, {png: maskPng()});
    maskDirty = false;
    $("emstate").textContent = r.mask ? `범위 저장 — 360 의 ${Math.round(r.cover * 1000) / 10}%` : "범위를 지웠어요";
    setMaskLink(r.mask);
    return r;
  }
  $("emsave").onclick = async () => {
    try { await saveMask(); } catch (e) { $("emstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
  function setMaskLink(url) {
    $("emdl").classList.toggle("hidden", !url);
    if (url) { $("emdl").href = url; $("emdl").setAttribute("download", `${runId}_a${String(n).padStart(2, "0")}_mask.png`); }
  }
  async function loadBase() {
    cx.clearRect(0, 0, cv.width, cv.height); maskDirty = false;
    try {
      const b = await api(`/api/run/${runId}/attempt/${n}/event/base`);
      $("embase").src = b.base;
      setMaskLink(b.mask);
      if (b.mask) {
        const im = new Image();
        im.onload = () => {                                    // 흰색 마스크 → 빨강 칠
          const t = document.createElement("canvas"); t.width = cv.width; t.height = cv.height;
          const tx = t.getContext("2d"); tx.drawImage(im, 0, 0, cv.width, cv.height);
          const d = tx.getImageData(0, 0, cv.width, cv.height), o = cx.createImageData(cv.width, cv.height);
          for (let i = 0; i < d.data.length; i += 4) { o.data[i] = 255; o.data[i + 1] = 40; o.data[i + 2] = 70; o.data[i + 3] = d.data[i]; }
          cx.putImageData(o, 0, 0);
        };
        im.src = b.mask;
        $("emstate").textContent = "저장된 범위를 불러왔어요";
      } else $("emstate").textContent = "움직일 곳을 칠하고 '범위 저장'을 누르세요";
    } catch (e) { $("emstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  }

  // ---- 360 전체 영상 프롬프트 (LTX + 360 LoRA 용) 복사
  let ltxDefaults = {};
  const copyTo = async (txt, what) => {
    try { await navigator.clipboard.writeText(txt); $("eltxstate").textContent = `${what} 를 복사했어요.`; }
    catch { $("eltxstate").textContent = "복사하지 못했어요. 칸에서 직접 복사하세요."; }
  };
  $("eltxcp").onclick = () => copyTo($("eltxpos").value, "Positive");
  $("eltxcn").onclick = () => copyTo($("eltxneg").value, "Negative");

  // ---- 영상 만들기: 마스크 인페인팅 노드 (화질 올린 360 + 칠한 범위 + 프롬프트)
  let vTimer = null;
  function vidShow(v) {
    clearTimeout(vTimer);
    $("evout").classList.toggle("hidden", !(v && v.state === "done"));
    if (!v || v.state === "none") { $("evstate").textContent = ""; $("evgo").disabled = false; return; }
    const nodeLink = v.node_url ? ` · <a href="${v.node_url}" download="${runId}_video_node.json">값 넣은 노드 파일 받기</a>` : "";
    const maskTxt = v.mask_mode === "rect" && v.rect ? ` · 마스크 사각형 x${v.rect[0]} y${v.rect[1]} ${v.rect[2]}×${v.rect[3]}` : v.mask_mode === "image" ? " · 마스크 그림" : "";
    if (v.state === "running") {
      const pr = v.progress || {};
      $("evgo").disabled = true;
      $("evstate").innerHTML = esc(`영상 만드는 중… ${pr.state === "running" ? (pr.node_title || pr.node || "") + (pr.max ? ` ${pr.value}/${pr.max}` : "") : "대기 중"}`) + esc(maskTxt) + nodeLink;
      vTimer = setTimeout(vidPoll, 5000);
      return;
    }
    $("evgo").disabled = false;
    if (v.state === "error") { $("evstate").innerHTML = `<span class="err">${esc(v.error || "실패")}</span>` + nodeLink; return; }
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
    try { vidShow(await api(`/api/run/${runId}/attempt/${n}/event/video`)); }
    catch (e) { $("evstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; vTimer = setTimeout(vidPoll, 8000); }
  }
  $("evgo").onclick = async () => {
    $("evgo").disabled = true; $("evstate").textContent = "범위·이미지를 올리고 작업을 보내는 중…";
    try {
      if (maskDirty) await saveMask();
      vidShow(await post(`/api/run/${runId}/attempt/${n}/event/video`, {
        backend: $("evbackend").value, seconds: +$("evsec").value, lora_strength: +$("evlora").value,
        positive: $("eltxpos").value, negative: $("eltxneg").value}));
    } catch (e) { $("evstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; $("evgo").disabled = false; }
  };

  return {
    show(id, att, cfg) {
      if (!att || att.state !== "done" || !(att.files || []).length) { this.hide(); return; }
      runId = id; n = att.n; ev = att.event || null;
      if (cfg && cfg.ltx360) ltxDefaults = cfg.ltx360;
      const lp = (ev && ev.ltx360) || ltxDefaults;
      $("eltxpos").value = lp.positive || ""; $("eltxneg").value = lp.negative || ""; $("eltxstate").textContent = "";
      loadBase();
      if (cfg && cfg.backends && !$("evbackend").options.length) {
        $("evbackend").innerHTML = cfg.backends.map(b =>
          `<option value="${esc(b.name)}">${esc(b.label)}${b.ok ? "" : " (연결 안 됨)"}</option>`).join("");
        const pref = cfg.backends.find(b => b.name === cfg.default_backend && b.ok) || cfg.backends.find(b => b.ok);
        if (pref) $("evbackend").value = pref.name;
      }
      const v = ev && ev.video;
      if (v && v.seconds) $("evsec").value = String(v.seconds);
      if (v && v.lora_strength != null) $("evlora").value = v.lora_strength;
      vidShow(v);
      $("eventcard").classList.remove("hidden");
      $("etarget").textContent = `#${att.n} 결과로 만들어요`;
      if (ev && ev.seconds) $("esec").value = String(Math.round(ev.seconds));
      $("estate").textContent = ev ? "저장된 사건을 불러왔어요." : "";
      render();
    },
    hide() { $("eventcard").classList.add("hidden"); ev = null; runId = null; clearTimeout(vTimer); },
  };
}
