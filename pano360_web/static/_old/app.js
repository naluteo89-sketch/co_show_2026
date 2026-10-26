import {api, post, esc} from "./api.js";
import {$, errHtml, usageText, modelOptions, fillBackends} from "./dom.js";
import {createViewer} from "./viewer.js";
import {createPoller} from "./poller.js";
import {setupMotion} from "./video.js";
import {setupEvent} from "./event.js";

let cfg = null, run = null, ctrlTimer = null;
const viewer = createViewer($("viewer"));
const motion = setupMotion();
const eventUI = setupEvent();
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
  motion.hide();
  eventUI.hide();
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
    $("model").innerHTML = modelOptions(cfg.openai_models);
    $("model").value = cfg.openai_model;
    $("chips").innerHTML = chip(cfg.openai_ready, `OpenAI · 기본 ${cfg.openai_model}`) +
      cfg.backends.map(b => chip(b.ok, b.label)).join("") +
      `<span class="chip">${esc(cfg.workflow)}</span>` +
      `<span class="chip" title="${esc(cfg.version)}">v ${esc((cfg.version || "?").split(" ").slice(0, 2).join(" "))}</span>`;
    fillBackends($("backend"), cfg);
    fillBackends($("backend2"), cfg);
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
  } catch (e) { if (!current(version)) return; $("srcinfo").innerHTML = errHtml(e.message); }
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
    $("anstate").textContent = usageText(r.model, r.usage);
  } catch (e) { if (!current(version)) return; $("anstate").innerHTML = errHtml(e.message); }
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

async function modelCheck() {
  const box = $("mcheckout"); box.classList.remove("hidden");
  box.innerHTML = `<div class="muted">점검 중… (${esc($("backend").selectedOptions[0]?.textContent || "")})</div>`;
  try {
    const r = await api(`/api/models/check?backend=${encodeURIComponent($("backend").value)}`);
    const mark = {ok: "✔", mapped: "✔", warn: "⚠", unknown: "?", missing: "✘"};
    box.innerHTML = r.groups.map(g => `<div class="mgroup"><b>${esc(g.title)}</b>` + (g.error ? `<div class="bad">${esc(g.error)}</div>` :
      g.items.map(i => `<div class="mi ${i.state}">${mark[i.state]} ${esc(i.class)} · ${esc(i.want)}` +
        (i.state === "mapped" ? ` <span class="muted">→ ${esc(i.use)}</span>` : "") +
        (i.state === "warn" ? ` → ${esc(i.use)}<br><span class="muted">${esc(i.note)}</span>` : "") +
        (i.state === "missing" || i.state === "unknown" ? `<br><span class="muted">${i.note ? linkify(i.note) : "서버에 없음" + (i.near?.length ? " · 비슷한 파일: " + esc(i.near.join(", ")) : "")}</span>` : "") +
        `</div>`).join("")) + `</div>`).join("");
  } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>`; }
}
const linkify = s => esc(s).replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

async function importAsResult() {
  if (!run) return;
  const id = run.run_id, version = viewVersion;
  $("importbtn").disabled = true; $("importstate").textContent = "등록 중…";
  try {
    const att = await post(`/api/run/${id}/import`, {});
    if (!current(version)) return;
    $("importstate").textContent = `#${att.n} 결과로 등록했어요 — 아래 5·6단계를 쓰세요.`;
    $("jobcard").classList.remove("hidden"); $("bar").style.width = "100%";
    $("jobtext").textContent = `#${att.n} 가져온 360 이미지 · ${(att.size || []).join("×")}`;
    showResult(att); loadAttempts(); loadRuns();
    $("jobcard").scrollIntoView({behavior: "smooth"});
  } catch (e) { if (current(version)) $("importstate").innerHTML = errHtml(e.message); }
  $("importbtn").disabled = false;
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
  const vq = x => x.v ? `?v=${x.v}` : "";
  $("results").innerHTML = shown.map(x => `<div>
    <div class="row" style="margin-bottom:8px">
      <a href="${(x.jpg || x.url) + vq(x)}" download style="flex:0 0 auto"><button class="primary">360 이미지 다운로드 (JPG)</button></a>
      <a href="${x.url + vq(x)}" download style="flex:0 0 auto"><button>원본 PNG</button></a>
      ${x.jpg ? `<a href="#" class="toviewer" data-n="${att.n}" style="flex:0 0 auto"><button>Quest 뷰어에 추가</button></a> <span class="tvstate muted"></span>` : ""}
    </div>
    <a href="${(x.jpg || x.url) + vq(x)}" target="_blank"><img src="${(x.jpg || x.url) + vq(x)}" alt="" class="pano"></a>
    <div class="muted dims" style="font-size:12px"></div></div>`).join("");
  const im = $("results").querySelector("img.pano");
  if (im) im.onload = () => { im.parentElement.nextElementSibling.textContent =
      `360 결과 ${im.naturalWidth} × ${im.naturalHeight}px` +
      " · 위 360 뷰어 화면은 미리보기라 저장해도 작은 화면 크기로만 저장돼요"; };
  if (f && att.mode !== "pole" && att.mode !== "enhance" && att.mode !== "import" && !String(att.photo_pasted || "").startsWith("upscaled"))
    $("results").insertAdjacentHTML("beforeend",
      `<div class="note">원본 사진 자리가 주변보다 흐리면 →
         <button id="repastebtn" data-n="${att.n}">원본 자리 화질 맞추기</button>
         <span class="muted" style="font-size:12px" id="repastestate">원본 사진을 원래 해상도로 AI 업스케일해서 같은 자리에 다시 붙여요 (1~2분, 이 결과를 고침)</span></div>`);
  if ($("repastebtn")) $("repastebtn").onclick = async () => {
    const b = $("repastebtn"); b.disabled = true; $("repastestate").textContent = "원본 사진 업스케일 중…";
    try {
      const r = await post(`/api/run/${run.run_id}/attempt/${b.dataset.n}/repaste`, {});
      $("repastestate").textContent = `완료 (${r.mode})`;
      showResult({...att, files: r.files, photo_pasted: r.mode});
      const nf = finalFile(r.files); if (nf) viewer.load((nf.jpg || nf.url) + (nf.v ? `?v=${nf.v}` : ""));
    } catch (e) { $("repastestate").innerHTML = errHtml(e.message); b.disabled = false; }
  };
  if (f) $("results").insertAdjacentHTML("beforeend",
    `<div class="note">발밑이 뭉개지거나 하늘 꼭대기가 비어 보이면 →
       <button id="polebtn" data-n="${att.n}">발밑·하늘 보정 + 8K</button>
       <span class="muted" style="font-size:12px">극점 부분만 다시 그려서 합성한 뒤 8K로 다시 만들어요 (새 시도로 저장)</span></div>`);
  if ($("polebtn")) $("polebtn").onclick = () => poleFix(+$("polebtn").dataset.n);
  document.querySelectorAll(".toviewer").forEach(el => el.onclick = async e => {
    e.preventDefault(); const st = el.nextElementSibling || el.parentElement.querySelector(".tvstate"); st.textContent = "복사 중…";
    try { const r = await post(`/api/run/${run.run_id}/attempt/${el.dataset.n}/to_viewer`, {});
          st.textContent = `추가됨 (사진 ${r.count}장) — Unity에서 다시 빌드하세요`; }
    catch (err) { st.innerHTML = errHtml(err.message); }
  });
  if (f) viewer.load(f.jpg || f.url);
  motion.show(run.run_id, att, cfg);       // 5. 영상화 프롬프트 (이 결과 기준)
  eventUI.show(run.run_id, att, cfg);           // 6. 사건 영상 (한 번 일어나는 연출)
}

async function loadAttempts() {
  if (!run) return;
  const version = viewVersion;
  const m = await api(`/api/run/${run.run_id}`);
  if (!current(version)) return;
  const fmtSet = s => s && Object.keys(s).length ? ` · steps ${s.steps ?? "-"} · CFG ${s.cfg ?? "-"} · LoRA ${s.lora_strength ?? "-"} · seam ${s.seam_denoise ?? "-"}` : "";
  $("attempts").innerHTML = m.attempts.slice().reverse().map(a => {
    const enh = a.mode === "enhance" || a.mode === "pole" || a.mode === "import";
    const head = a.mode === "import" ? `#${a.n} · 가져온 360 이미지 (생성 없음) · ${a.created}`
               : enh ? `#${a.n} · ${a.backend} · ${a.state} · ${a.mode === "pole" ? `발밑·하늘 보정${a.from_attempt ? ` (#${a.from_attempt} 결과)` : ""} + 8K` : "8K 화질 개선"} · ${a.created}`
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
$("importbtn").onclick = importAsResult;
$("mcheck").onclick = modelCheck;
$("polebtn2").onclick = () => poleFix(null);
init();
