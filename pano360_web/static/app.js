import {api, post, esc} from "./api.js";
import {createViewer} from "./viewer.js";
import {createPoller} from "./poller.js";

const $ = id => document.getElementById(id);
let cfg = null, run = null, ctrlTimer = null;
const viewer = createViewer($("viewer"));
let viewVersion = 0, controlVersion = 0, sourceUrl = null;
const poller = createPoller(
  (id, n, signal) => api(`/api/run/${id}/attempt/${n}`, {signal}),
  showStatus,
  error => { $("jobtext").textContent = error.message; }
);

function setBusy(busy) {
  ["gen", "enhance", "polebtn", "polebtn2"].forEach(id => { if ($(id)) $(id).disabled = busy; });
}

function resetView() {
  viewVersion++;
  controlVersion++;
  poller.stop();
  clearTimeout(ctrlTimer);
  viewer.hide();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = null;
  run = null;
  setBusy(false);
  $("analyze").disabled = false;
  ["anacard", "promptcard", "jobcard", "erpbox"].forEach(id => $(id).classList.add("hidden"));
  $("anstate").textContent = "";
  return viewVersion;
}

const current = version => version === viewVersion;
const numberValue = id => $(id).value.trim() === "" ? null : Number($(id).value);
const settings = () => ({steps: numberValue("g_steps"), cfg: numberValue("g_cfg"),
  lora_strength: numberValue("g_lora"), seam_denoise: numberValue("g_seam")});
const finalFile = files => files.find(f => f.final) || files.find(f => ["8k", "4k"].includes(f.tag)) || files[0];

async function init() {
  try {
    cfg = await api("/api/config");
    const chip = (ok, t) => `<span class="chip ${ok ? "ok" : "bad"}">${esc(t)}</span>`;
    $("model").innerHTML = cfg.openai_models.map(m => `<option>${esc(m)}</option>`).join("");
    $("model").value = cfg.openai_model;
    $("chips").innerHTML = chip(cfg.openai_ready, `OpenAI · 기본 ${cfg.openai_model}`) +
      cfg.backends.map(b => chip(b.ok, b.label)).join("") +
      `<span class="chip">${esc(cfg.workflow)}</span>` +
      `<span class="chip" title="${esc(cfg.version)}">v ${esc((cfg.version || "?").split(" ").slice(0, 2).join(" "))}</span>`;
    $("backend").innerHTML = cfg.backends.map(b =>
      `<option value="${b.name}">${esc(b.label)}${b.ok ? "" : " (연결 안 됨)"}</option>`).join("");
    $("backend2").innerHTML = $("backend").innerHTML;
    const first = cfg.backends.find(b => b.ok); if (first) { $("backend").value = first.name; $("backend2").value = first.name; }
    const g = cfg.gen_defaults || {};
    $("g_steps").value = g.steps ?? ""; $("g_cfg").value = g.cfg ?? ""; $("g_lora").value = g.lora_strength ?? ""; $("g_seam").value = g.seam_denoise ?? "";
  } catch (e) { $("chips").innerHTML = `<span class="chip bad">서버 연결 실패</span>`; }
  loadRuns();
}

async function loadRuns() {
  try {
    const rs = await api("/api/runs");
    $("runs").innerHTML = rs.length ? rs.map(r => `
      <div class="r" data-id="${r.run_id}">
        <img src="${r.last && r.last.files[0] ? r.last.files[0].url : r.thumb}" alt="">
        <div class="t"><div>${esc(r.name || r.run_id)}</div><div class="muted">${r.run_id.slice(0, 15)} · ${r.attempts}회</div></div>
      </div>`).join("") : "아직 없어요";
    document.querySelectorAll(".runs .r").forEach(el => el.onclick = () => openRun(el.dataset.id));
  } catch (e) { $("runs").textContent = "목록 실패"; }
}

// ---------- 1. 업로드
async function upload(file) {
  const version = resetView();
  $("srcbox").classList.remove("hidden");
  sourceUrl = URL.createObjectURL(file);
  $("srcimg").src = sourceUrl;
  $("srcinfo").textContent = "업로드 중…";
  ["anacard", "promptcard", "jobcard"].forEach(id => $(id).classList.add("hidden"));
  const fd = new FormData(); fd.append("image", file);
  try {
    const uploaded = await api("/api/upload", {method: "POST", body: fd});
    if (!current(version)) return;
    run = uploaded;
    $("srcinfo").innerHTML = `${run.width} × ${run.height}px` + (run.is_erp ? " · 360 이미지" : (run.exif_hfov ? ` · EXIF 화각 ${run.exif_hfov}°` : " · EXIF 화각 없음 (LLM이 추정)"));
    $("erpbox").classList.toggle("hidden", !run.is_erp);
    loadRuns();
  } catch (e) { if (!current(version)) return; $("srcinfo").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

// ---------- 2. 분석
async function analyze() {
  if (!run) return;
  const version = viewVersion;
  $("analyze").disabled = true; $("anstate").textContent = "분석 중… (10~40초)";
  try {
    const r = await post("/api/analyze", {run_id: run.run_id, note: $("note").value, model: $("model").value});
    if (!current(version)) return;
    showAnalysis(r.analysis, r.placement, r.prompts);
    const u = r.usage || {};
    $("anstate").textContent = `${r.model} · 입력 ${u.input_tokens ?? "?"} / 출력 ${u.output_tokens ?? "?"} 토큰`;
  } catch (e) { if (!current(version)) return; $("anstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("analyze").disabled = false;
}

function showAnalysis(a, pl, pr) {
  $("anacard").classList.remove("hidden"); $("promptcard").classList.remove("hidden");
  $("summary").textContent = a.summary_ko; $("tips").textContent = "주의 · " + a.tips_ko;
  $("yaw").value = pl.yaw; $("pitch").value = pl.pitch; $("hfov").value = pl.hfov; $("vfov").value = pl.vfov ?? "";
  $("hfovsrc").textContent = pl.hfov_source ? `· ${pl.hfov_source}` : "";
  $("p_pos").value = pr.positive; $("p_neg").value = pr.negative;
  $("p_spos").value = pr.seam_positive; $("p_sneg").value = pr.seam_negative;
  $("raw").textContent = JSON.stringify(a, null, 2);
  refreshControl();
}

const placement = () => ({yaw: +$("yaw").value, pitch: +$("pitch").value, hfov: +$("hfov").value,
                          vfov: $("vfov").value === "" ? null : +$("vfov").value});
function refreshControl() {
  clearTimeout(ctrlTimer);
  if (!run) return;
  const id = run.run_id, version = viewVersion, request = ++controlVersion;
  const pl = placement();
  ctrlTimer = setTimeout(async () => {
    try {
      const result = await post("/api/control", {run_id: id, placement: pl});
      if (current(version) && request === controlVersion) $("ctrlimg").src = result.url;
    } catch (e) { console.error(e); }
  }, 300);
}

// ---------- 생성·화질 개선·극점 보정 공통 제출
async function submitJob(body, message, scroll = false) {
  if (!run) return;
  const id = run.run_id, version = viewVersion;
  setBusy(true);
  try {
    const result = await post("/api/generate", {run_id: id, ...body});
    if (!current(version)) return;
    $("jobcard").classList.remove("hidden");
    $("results").innerHTML = "";
    viewer.hide();
    $("bar").style.width = "0%";
    $("jobtext").textContent = `#${result.attempt} ${message(result)}`;
    poller.start(id, result.attempt);
    if (scroll) $("jobcard").scrollIntoView({behavior: "smooth"});
  } catch (error) {
    if (!current(version)) return;
    alert(error.message);
    setBusy(false);
  }
}

function generate() {
  if (!$("p_pos").value.startsWith("Fill the green spaces"))
    if (!confirm("Positive 프롬프트가 LoRA 트리거 문장으로 시작하지 않아요. 그대로 진행할까요?")) return;
  return submitJob({backend: $("backend").value, placement: placement(),
    seed1: numberValue("seed1"), seed2: numberValue("seed2"), settings: settings(),
    prompts: {positive: $("p_pos").value, negative: $("p_neg").value,
      seam_positive: $("p_spos").value, seam_negative: $("p_sneg").value}},
    r => `대기 중… (seed ${r.seeds.seed1} / ${r.seeds.seed2})`);
}

function poleFix(fromN) {
  const backend = fromN ? $("backend").value : $("backend2").value;
  return submitJob({backend, mode: "pole", from_attempt: fromN || null, settings: settings()},
    () => "발밑·하늘 보정 대기 중…", true);
}

function enhance() {
  return submitJob({backend: $("backend2").value, mode: "enhance"}, () => "화질 개선 대기 중…");
}

function showStatus(s, n) {
  if (s.state === "queued") $("jobtext").textContent = `#${n} 대기열에서 기다리는 중…`;
  else if (s.state === "running") {
    $("jobtext").textContent = `#${n} ${s.node_title || "실행"} 중… ${s.max ? `${s.value}/${s.max}` : ""}`;
    $("bar").style.width = (s.max ? Math.round(s.value / s.max * 100) : 5) + "%";
  } else if (s.state === "done") {
    setBusy(false); $("bar").style.width = "100%";
    $("jobtext").textContent = `#${n} 완료 · ${s.finished || ""}`;
    showResult(s); loadAttempts(); loadRuns();
  } else if (s.state === "error") {
    setBusy(false);
    $("jobtext").textContent = `#${n} 실패: ${s.error}`;
    loadAttempts();
  }
}

function showResult(att) {
  // 결과는 최종 360 한 장만 보여줌 (예전 시도에 남은 2K 등 중간 파일은 숨김)
  const f = finalFile(att.files);
  const shown = f ? [f] : [];
  $("results").innerHTML = shown.map(x => `<div>
    <div class="row" style="margin-bottom:8px">
      <a href="${x.jpg || x.url}" download style="flex:0 0 auto"><button class="primary">360 이미지 다운로드 (JPG)</button></a>
      <a href="${x.url}" download style="flex:0 0 auto"><button>원본 PNG</button></a>
      ${x.jpg ? `<a href="#" class="toviewer" data-n="${att.n}" style="flex:0 0 auto"><button>Quest 뷰어에 추가</button></a> <span class="tvstate muted"></span>` : ""}
    </div>
    <a href="${x.jpg || x.url}" target="_blank"><img src="${x.jpg || x.url}" alt="" class="pano"></a>
    <div class="muted dims" style="font-size:12px"></div></div>`).join("");
  const im = $("results").querySelector("img.pano");
  if (im) im.onload = () => { im.parentElement.nextElementSibling.textContent =
      `360 결과 ${im.naturalWidth} × ${im.naturalHeight}px` +
      (att.seam ? ` · 좌우 이음새 ${att.seam.ratio <= 2 ? "양호" : att.seam.ratio <= 3.5 ? "약간 보임" : "눈에 띔 — 다시 생성 권장"} (×${att.seam.ratio})` : "") +
      " · 위 360 뷰어 화면은 미리보기라 저장해도 작은 화면 크기로만 저장돼요"; };
  if (f) $("results").insertAdjacentHTML("beforeend",
    `<div class="note">발밑이 뭉개지거나 하늘 꼭대기가 비어 보이면 →
       <button id="polebtn" data-n="${att.n}">발밑·하늘 보정 + 8K</button>
       <span class="muted" style="font-size:12px">극점 부분만 다시 그려서 합성한 뒤 8K로 다시 만들어요 (새 시도로 저장)</span></div>`);
  if ($("polebtn")) $("polebtn").onclick = () => poleFix(+$("polebtn").dataset.n);
  document.querySelectorAll(".toviewer").forEach(el => el.onclick = async e => {
    e.preventDefault(); const st = el.nextElementSibling || el.parentElement.querySelector(".tvstate"); st.textContent = "복사 중…";
    try { const r = await post(`/api/run/${run.run_id}/attempt/${el.dataset.n}/to_viewer`, {});
          st.textContent = `추가됨 (사진 ${r.count}장) — Unity에서 다시 빌드하세요`; }
    catch (err) { st.innerHTML = `<span class="err">${esc(err.message)}</span>`; }
  });
  if (f) viewer.load(f.jpg || f.url);
}

async function loadAttempts() {
  if (!run) return;
  const version = viewVersion;
  const m = await api(`/api/run/${run.run_id}`);
  if (!current(version)) return;
  const fmtSet = s => s && Object.keys(s).length ? ` · steps ${s.steps ?? "-"} · CFG ${s.cfg ?? "-"} · LoRA ${s.lora_strength ?? "-"} · seam ${s.seam_denoise ?? "-"}` : "";
  $("attempts").innerHTML = m.attempts.slice().reverse().map(a => {
    const enh = a.mode === "enhance" || a.mode === "pole";
    const head = enh ? `#${a.n} · ${a.backend} · ${a.state} · ${a.mode === "pole" ? `발밑·하늘 보정${a.from_attempt ? ` (#${a.from_attempt} 결과)` : ""} + 8K` : "8K 화질 개선"} · ${a.created}`
                     : `#${a.n} · ${a.backend} · ${a.state} · seed ${a.seeds.seed1}/${a.seeds.seed2}${fmtSet(a.settings)} · ${a.created}`;
    return `<details class="attempt"><summary>${esc(head)}</summary>
      ${[finalFile(a.files)].filter(Boolean).map(f => `<a href="#" data-url="${f.jpg || f.url}" class="show" style="color:var(--acc);margin-right:10px">360 보기</a>`).join("")}
      ${enh ? "" : `<a href="#" data-n="${a.n}" class="reuse" style="color:var(--acc)">이 설정 다시 불러오기</a>
      <pre>${esc(a.prompts.positive)}\n\nNEG: ${esc(a.prompts.negative)}\n\nyaw ${a.placement.yaw} · pitch ${a.placement.pitch} · hFOV ${a.placement.hfov}</pre>`}
      ${a.error ? `<div class="err">${esc(a.error)}</div>` : ""}
    </details>`;
  }).join("");
  document.querySelectorAll(".attempt .show").forEach(el => el.onclick = e => { e.preventDefault(); viewer.load(el.dataset.url); });
  document.querySelectorAll(".attempt .reuse").forEach(el => el.onclick = e => {
    e.preventDefault(); const a = m.attempts[+el.dataset.n - 1];
    showAnalysis(m.analysis, a.placement, a.prompts); $("seed1").value = a.seeds.seed1; $("seed2").value = a.seeds.seed2;
    const s = a.settings || {};
    if (s.steps) $("g_steps").value = s.steps; if (s.cfg) $("g_cfg").value = s.cfg;
    if (s.lora_strength != null) $("g_lora").value = s.lora_strength; if (s.seam_denoise) $("g_seam").value = s.seam_denoise;
    $("promptcard").scrollIntoView({behavior: "smooth"});
  });
  return m;
}

async function openRun(id) {
  const version = resetView();
  let m;
  try { m = await api(`/api/run/${id}`); }
  catch (error) { if (current(version)) alert(error.message); return; }
  if (!current(version)) return;
  run = m;
  $("srcbox").classList.remove("hidden"); $("srcimg").src = `/runs/${id}/source${m.source_ext}`;
  $("srcinfo").textContent = `${m.width} × ${m.height}px · ${m.original_name || ""}`;
  $("note").value = m.note || "";
  if (m.model) $("model").value = m.model;
  $("erpbox").classList.toggle("hidden", !m.is_erp);
  if (m.analysis) {
    const last = m.attempts.slice().reverse().find(a => a.placement?.hfov && a.prompts?.positive);
    showAnalysis(m.analysis, last ? last.placement : m.placement, last ? last.prompts : m.prompts);
  } else ["anacard", "promptcard"].forEach(x => $(x).classList.add("hidden"));
  if (m.attempts.length) {
    $("jobcard").classList.remove("hidden"); $("bar").style.width = "0";
    const done = m.attempts.slice().reverse().find(a => a.files.length);
    $("results").innerHTML = ""; viewer.hide(); $("jobtext").textContent = "";
    if (done) showResult(done);
    const pending = m.attempts.find(a => a.state !== "done" && a.state !== "error");
    if (pending) { setBusy(true); poller.start(id, pending.n); }
    loadAttempts();
  } else $("jobcard").classList.add("hidden");
  window.scrollTo({top: 0, behavior: "smooth"});
}

// ---------- 이벤트
$("drop").onclick = () => $("file").click();
$("file").onchange = e => e.target.files[0] && upload(e.target.files[0]);
["dragenter", "dragover"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
$("analyze").onclick = analyze;
["yaw", "pitch", "hfov", "vfov"].forEach(id => $(id).addEventListener("input", refreshControl));
$("gen").onclick = generate;
$("enhance").onclick = enhance;
$("polebtn2").onclick = () => poleFix(null);
init();
