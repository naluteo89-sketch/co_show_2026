// ---------- 영상화: 움직일 곳(일부만) + 영상 프롬프트 제작기
// 결과 360 과 생성 프롬프트를 LLM 에 보내 '어디를 어떻게 움직일지' 받고, 영역별 Cinemagraph 프롬프트를 만든다.
// 영상은 ComfyUI/LTX 쪽에서 만든다 — 여기서는 영역(마스크)과 프롬프트만.

import {api, post, esc} from "./api.js";
import {$, errHtml, usageText, modelOptions, copyText} from "./dom.js";

const FIELDS = ["name_ko", "element", "yaw", "pitch", "hfov", "aspect", "motion", "speed", "strength", "keep_still",
                "spread", "spread_strength"];
const NUM = new Set(["yaw", "pitch", "hfov", "aspect", "strength", "spread_strength"]);
const aspectOf = r => r.aspect || (r.orientation === "portrait" ? 512 / 704 : 704 / 512);
function sizeOf(a) {                                       // 서버 studio/erp/regions.py region_size 와 같은 계산
  a = Math.min(2, Math.max(0.5, a));
  const w = Math.sqrt(1408 * 1024 * a), h = w / a;
  return [Math.max(384, Math.round(w / 64) * 64), Math.max(384, Math.round(h / 64) * 64)];
}
const COLORS = ["#ff5a6e", "#5ac8ff", "#ffc83c", "#78e68c", "#c88cff"];

export function setupMotion() {
  let runId = null, n = null, plan = null, models = [], pickTarget = null;

  function regionHtml(r, i) {
    const v = k => esc(r[k] ?? "");
    return `<div class="mregion" data-i="${i}" style="border-left:4px solid ${COLORS[i % COLORS.length]}">
      <div class="mhead"><b>${i + 1}. ${esc(r.name_ko || "영역")}</b>
        <span class="muted">${esc(r.risk_ko || "")}</span>
        <button class="mpos" title="위 360 화면에서 마우스로 끌어 이 영역 위치를 다시 고르기">위치 다시 고르기</button>
        <button class="mdel" title="이 영역 빼기">빼기</button></div>
      <div class="mgrid">
        <div class="mcrop">${r.crop ? `<a href="${r.crop}" target="_blank"><img src="${r.crop}" alt=""></a>
          <div class="muted" style="font-size:12px">영상 모델에 들어갈 화면 · ${r.crop_size ? r.crop_size.join("×") : ""}</div>` : ""}</div>
        <div>
          <div class="row">
            <div><label>이름</label><input data-k="name_ko" value="${v("name_ko")}"></div>
            <div><label>움직이는 것 (영어)</label><input data-k="element" value="${v("element")}"></div>
          </div>
          <div class="row">
            <div><label>yaw (°)</label><input data-k="yaw" type="number" step="1" value="${v("yaw")}"></div>
            <div><label>pitch (°)</label><input data-k="pitch" type="number" step="1" value="${v("pitch")}"></div>
            <div><label>가로 화각 (°)</label><input data-k="hfov" type="number" step="1" min="20" max="120" value="${v("hfov")}"></div>
            <div><label>비율 (가로÷세로) <span class="muted masz">${sizeOf(aspectOf(r)).join("×")}</span></label>
              <input data-k="aspect" type="number" step="0.05" min="0.5" max="2" value="${Number(aspectOf(r)).toFixed(2)}"></div>
          </div>
          <div class="row">
            <div style="flex:3 1 300px"><label>움직임 (영어)</label><input data-k="motion" value="${v("motion")}"></div>
            <div><label>속도</label><select data-k="speed">${["subtle", "slow", "medium"].map(s =>
              `<option${r.speed === s ? " selected" : ""}>${s}</option>`).join("")}</select></div>
            <div><label>LoRA 강도</label><input data-k="strength" type="number" step="0.1" min="0.5" max="3" value="${v("strength")}"></div>
          </div>
          <label>이 화면 안에서 멈춰 있어야 할 것 (영어)</label><input data-k="keep_still" value="${v("keep_still")}">
          <label style="margin-top:8px">영역 화면용 Positive <span class="muted">(위 칸을 고치면 '수정 반영' 때 다시 만들어요)</span></label>
          <textarea data-p="positive" style="min-height:96px">${v("positive")}</textarea>
          <label style="margin-top:8px">Negative</label>
          <textarea data-p="negative" style="min-height:60px">${v("negative")}</textarea>
          <div class="minf">
            <div class="row mspread">
              <div><label>퍼짐 <span class="muted">— 요소 모양 밖으로</span></label><select data-k="spread">${[["none", "끔 (요소 모양 안만)"], ["near", "주변 (~35°)"], ["wide", "넓게 (~100°)"]].map(([k, t]) =>
                `<option value="${k}"${(r.spread || "near") === k ? " selected" : ""}>${t}</option>`).join("")}</select></div>
              <div style="flex:0 0 120px"><label>퍼짐 세기</label><input data-k="spread_strength" type="number" min="0" max="1" step="0.1" value="${esc(r.spread_strength ?? 0.7)}"></div>
            </div>
            <label>함께 움직일 것 <span class="muted">(선택) — 이 움직임 때문에 같이 잔잔히 움직일 것 (360 전체 프롬프트에 들어가요)</span></label>
            ${(r.influence || []).map((x, j) => `<div class="minfrow" data-j="${j}">
              <input data-f="name_ko" value="${esc(x.name_ko ?? "")}" placeholder="이름" style="flex:1 1 90px">
              <input data-f="target" value="${esc(x.target ?? "")}" placeholder="대상 (영어, 예: stream water below the falls)" style="flex:3 1 220px">
              <select data-f="range" title="범위">${[["none", "끔"], ["near", "주변 (~35°)"], ["wide", "넓게 (~100°)"]].map(([k, t]) =>
                `<option value="${k}"${(x.range || "near") === k ? " selected" : ""}>${t}</option>`).join("")}</select>
              <input data-f="strength" type="number" min="0" max="1" step="0.1" value="${esc(x.strength ?? 0.5)}" title="세기 0~1" style="flex:0 0 70px">
              <button class="minfdel" title="빼기">×</button>
              ${x.reason_ko ? `<div class="muted" style="flex:1 1 100%;font-size:12px">${esc(x.reason_ko)}</div>` : ""}
            </div>`).join("")}
            <button class="minfadd">영향 추가</button>
          </div>
        </div>
      </div></div>`;
  }

  function render() {
    $("mbody").classList.toggle("hidden", !plan);
    if (!plan) return;
    $("msummary").innerHTML = `<b>움직임</b> · ${esc(plan.summary_ko || "")}<br><b>정지</b> · ${esc(plan.static_ko || "")}`;
    if ($("moverlay").getAttribute("src") !== (plan.overlay || "")) { $("moverlay").src = plan.overlay || ""; $("mdrag").classList.add("hidden"); }
    $("mregions").innerHTML = (plan.regions || []).map(regionHtml).join("") ||
      `<div class="muted">움직일 영역이 없어요. '영역 추가'로 직접 넣을 수 있어요.</div>`;
    document.querySelectorAll("#mregions .mregion").forEach(el => {
      const i = +el.dataset.i;
      el.querySelectorAll("[data-k]").forEach(inp => inp.addEventListener("input", () => { plan.regions[i].regen = true; }));
      const asp = el.querySelector('[data-k="aspect"]');
      asp.addEventListener("input", () => { el.querySelector(".masz").textContent = sizeOf(Number(asp.value) || 1).join("×"); });
      el.querySelectorAll("[data-p]").forEach(inp => inp.addEventListener("input", () => { plan.regions[i].regen = false; }));
      el.querySelector(".mdel").onclick = () => { collect(); plan.regions.splice(i, 1); render(); };
      el.querySelector(".minfadd").onclick = () => { collect();
        (plan.regions[i].influence = plan.regions[i].influence || []).push({name_ko: "", target: "", range: "near", strength: 0.5, reason_ko: ""});
        render(); };
      el.querySelectorAll(".minfrow").forEach(row => row.querySelector(".minfdel").onclick = () => {
        collect(); plan.regions[i].influence.splice(+row.dataset.j, 1); render(); });
      el.classList.toggle("picking", pickTarget === i);
      el.querySelector(".mpos").onclick = () => { pickTarget = pickTarget === i ? null : i; render();
        if (pickTarget !== null) $("mpick").scrollIntoView({behavior: "smooth", block: "center"}); };
    });
    $("mpickmode").textContent = pickTarget !== null && plan.regions[pickTarget]
      ? `— 지금 끌면 ${pickTarget + 1}번 영역 위치가 바뀌어요` : "";
    const lt = plan.ltx || {};
    const trig = /^CINEMAGRAPH_MOTION\.\s*/;
    $("mltxpos").value = $("mltxfor").value === "ltx" ? (lt.positive || "") : (lt.positive || "").replace(trig, "");
    $("mltxneg").value = lt.negative || "";
    $("mltxnote").textContent = lt.positive ? "필요하면 고쳐서 복사하세요." : "예전 계획이에요 — '수정 반영'을 누르면 360 전체 프롬프트가 만들어져요.";
    const vm = plan.video_model || {};
    $("mmeta").innerHTML = esc(`${plan.model || ""} · ${plan.updated || plan.created || ""}`) +
      (plan.mask ? ` · <a href="${plan.mask}" download="motion_mask.png">영역 마스크 PNG 받기</a> (360 크기, 흰색 = 움직일 곳)` : "");
  }

  function collect() {
    document.querySelectorAll("#mregions .mregion").forEach(el => {
      const r = plan.regions[+el.dataset.i];
      el.querySelectorAll("[data-k]").forEach(inp => { r[inp.dataset.k] = NUM.has(inp.dataset.k) ? Number(inp.value) : inp.value; });
      el.querySelectorAll("[data-p]").forEach(inp => { r[inp.dataset.p] = inp.value; });
      r.influence = [...el.querySelectorAll(".minfrow")].map(row => {
        const old = (r.influence || [])[+row.dataset.j] || {};
        const g = k => row.querySelector(`[data-f="${k}"]`).value;
        return {...old, name_ko: g("name_ko"), target: g("target"), range: g("range"), strength: Number(g("strength"))};
      }).filter(x => x.target.trim());
    });
  }

  async function runPlan() {
    $("mplan").disabled = true; $("mstate").textContent = "LLM 이 결과 360 을 보고 움직일 곳을 고르는 중… (20~60초)";
    try {
      plan = await post(`/api/run/${runId}/attempt/${n}/motion`, {note: $("mnote").value, model: $("mmodel").value});
      $("mstate").textContent = usageText(plan.model, plan.usage);
      render();
    } catch (e) { $("mstate").innerHTML = errHtml(e.message); }
    $("mplan").disabled = false;
  }

  async function save() {
    collect();
    const empty = (plan.regions || []).findIndex(r => !String(r.element || "").trim() || !String(r.motion || "").trim());
    if (empty >= 0) { $("mstate").innerHTML = `<span class="err">${empty + 1}번 영역의 '움직이는 것'과 '움직임'(영어)을 채워 주세요.</span>`; return; }
    $("msave").disabled = true; $("mstate").textContent = "영역 화면과 미리보기를 다시 만드는 중…";
    try {
      plan = await post(`/api/run/${runId}/attempt/${n}/motion/update`, {regions: plan.regions, lighting: plan.lighting});
      $("mstate").textContent = "저장했어요.";
      render();
    } catch (e) { $("mstate").innerHTML = errHtml(e.message); }
    $("msave").disabled = false;
  }

  $("mplan").onclick = runPlan;
  $("msave").onclick = save;
  // ---- 마우스로 영역 고르기: 360(ERP) 미리보기 위에서 끌면 사각형 → yaw/pitch/화각/방향
  const LAND = 1408 / 1024;                                   // 영역 화면 가로:세로
  function rectToRegion(x0, y0, x1, y1) {                     // 0~1 좌표 → 끈 모양 그대로(자유 비율)
    const [ax, bx] = [Math.min(x0, x1), Math.max(x0, x1)], [ay, by] = [Math.min(y0, y1), Math.max(y0, y1)];
    const yaw = Math.round(((ax + bx) / 2 - 0.5) * 360);
    const pitch = Math.round((0.5 - (ay + by) / 2) * 180);
    const w = Math.max(5, (bx - ax) * 360 * Math.cos(pitch * Math.PI / 180));   // 실제 가로 각도 (위도만큼 줄어듦)
    const h = Math.max(5, (by - ay) * 180);
    const rad = Math.PI / 360;
    let aspect = Math.tan(w * rad) / Math.tan(h * rad);       // 평면 화면에서의 가로:세로
    aspect = Math.min(2, Math.max(0.5, aspect));
    // 비율을 2 이내로 맞추면서도 끈 사각형이 다 들어가게 화각을 정함 (+10% 여유)
    const hNeed = 2 * Math.atan(Math.tan(h * rad) * aspect) * 180 / Math.PI;
    const hfov = Math.round(Math.min(120, Math.max(20, Math.max(w, hNeed) * 1.1)));
    return {yaw, pitch: Math.max(-70, Math.min(70, pitch)), hfov, aspect: Math.round(aspect * 100) / 100,
            orientation: aspect < 1 ? "portrait" : "landscape"};
  }
  let drag = null;
  const pick = $("mpick");
  const rel = e => { const b = pick.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)), Math.min(1, Math.max(0, (e.clientY - b.top) / b.height))]; };
  function drawDrag(a, b, label) {
    const d = $("mdrag");
    d.classList.remove("hidden");
    d.style.left = `${Math.min(a[0], b[0]) * 100}%`; d.style.top = `${Math.min(a[1], b[1]) * 100}%`;
    d.style.width = `${Math.abs(a[0] - b[0]) * 100}%`; d.style.height = `${Math.abs(a[1] - b[1]) * 100}%`;
    d.innerHTML = `<span>${esc(label)}</span>`;
  }
  pick.addEventListener("pointerdown", e => {
    if (!plan || e.button !== 0) return;
    e.preventDefault(); pick.setPointerCapture(e.pointerId);
    drag = {a: rel(e), b: rel(e)};
  });
  pick.addEventListener("pointermove", e => {
    if (!drag) return;
    drag.b = rel(e);
    const r = rectToRegion(...drag.a, ...drag.b);
    drawDrag(drag.a, drag.b, `yaw ${r.yaw}° · pitch ${r.pitch}° · 화각 ${r.hfov}° · ${sizeOf(r.aspect).join("×")}`);
  });
  pick.addEventListener("pointerup", e => {
    if (!drag) return;
    const {a, b} = drag; drag = null;
    if (Math.abs(a[0] - b[0]) < 0.01 || Math.abs(a[1] - b[1]) < 0.01) { $("mdrag").classList.add("hidden"); return; }
    collect();
    const r = rectToRegion(...a, ...b);
    if (pickTarget !== null && plan.regions[pickTarget]) {
      Object.assign(plan.regions[pickTarget], r, {regen: true});
      $("mstate").textContent = `${pickTarget + 1}번 영역 위치를 바꿨어요. '수정 반영'을 누르면 영역 화면을 다시 만들어요.`;
      pickTarget = null;
    } else {
      plan.regions = plan.regions || [];
      plan.regions.push({name_ko: `영역 ${plan.regions.length + 1}`, element: "", ...r, motion: "",
                         speed: "slow", strength: 1.2, keep_still: "", regen: true});
      $("mstate").textContent = `새 영역을 넣었어요. '움직이는 것'과 '움직임'(영어)을 채우고 '수정 반영'을 눌러 주세요.`;
    }
    render();
  });
  pick.addEventListener("pointercancel", () => { drag = null; $("mdrag").classList.add("hidden"); });

  $("madd").onclick = () => {
    if (!plan) plan = {regions: [], summary_ko: "", static_ko: "", lighting: ""};
    collect();
    plan.regions.push({name_ko: "새 영역", element: "", yaw: 0, pitch: 0, hfov: 60, aspect: 1.38, orientation: "landscape",
                       motion: "", speed: "slow", strength: 1.2, keep_still: "", regen: true});
    render();
  };
  const copyLtx = (txt, what) => copyText(txt, $("mltxnote"), `${what} 를 복사했어요.`,
                                          "복사하지 못했어요. 칸에서 직접 선택해 복사하세요.");
  $("mltxfor").onchange = () => { if (plan) { collect(); render(); } };
  $("mltxcp").onclick = () => copyLtx($("mltxpos").value, "Positive");
  $("mltxcn").onclick = () => copyLtx($("mltxneg").value, "Negative");
  $("mcopy").onclick = async () => {
    collect();
    const out = {lighting: plan.lighting, video_model: plan.video_model,
      panorama: {positive: $("mltxpos").value, negative: $("mltxneg").value}, regions: plan.regions.map(r =>
      Object.fromEntries([...FIELDS, "positive", "negative", "crop", "crop_size"].map(k => [k, r[k]])))};
    await copyText(JSON.stringify(out, null, 2), $("mstate"), "프롬프트 JSON 을 복사했어요.",
                   "복사하지 못했어요. 작업 폴더의 motion/plan.json 을 쓰세요.");
  };

  return {
    show(id, att, cfg) {
      if (!att || att.state !== "done" || !(att.files || []).length) { this.hide(); return; }
      runId = id; n = att.n; plan = att.motion || null;
      if (cfg && cfg.openai_models && !models.length) {
        models = cfg.openai_models;
        $("mmodel").innerHTML = modelOptions(models);
        $("mmodel").value = cfg.openai_model;
      }
      $("motioncard").classList.remove("hidden");
      $("mtarget").textContent = `#${att.n} 결과로 만들어요`;
      $("mstate").textContent = plan ? "저장된 영상화 계획을 불러왔어요." : "";
      render();
    },
    hide() { $("motioncard").classList.add("hidden"); plan = null; runId = null; },
  };
}
