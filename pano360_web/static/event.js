// ---------- 6. 사건 영상: 한 번 일어나는 연출 (잔잔 → 물이 넘쳐 화면을 덮음 → 다시 잔잔)
// LLM 이 사건 위치·내용·프롬프트를 정하고, 사건 화면(16:9) + ComfyUI 워크플로를 준다.
// ComfyUI 에서 만든 MP4 를 올리면 원본에서 달라진 곳만 360 위에 섞어 MP4 로.

const KEYS = ["yaw", "pitch", "hfov", "calm_start", "event", "peak", "calm_end", "keep"];
const NUM = new Set(["yaw", "pitch", "hfov"]);

export function setupEvent({api, post, esc}) {
  const $ = id => document.getElementById(id);
  let runId = null, n = null, ev = null, edited = false;

  function render() {
    $("ebody").classList.toggle("hidden", !(ev && ev.view));
    showResult(ev && ev.result);
    if (!ev || !ev.view) return;
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
    $("einfo").innerHTML = esc(`${(r.size || []).join("×")} · ${r.fps}fps · ${r.seconds}초` +
      (r.layers ? ` · 레이어 ${r.layers}개` : "") +
      (r.peak_cover != null ? ` · 가장 많이 덮인 때 ${Math.round(r.peak_cover * 100)}% (시작 ${Math.round((r.start_cover || 0) * 100)}% · 끝 ${Math.round((r.end_cover || 0) * 100)}%)` : "") +
      (r.max_shift_px != null ? ` · 흔들림 보정 최대 ${r.max_shift_px}px` : "")) +
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

  // ---- 효과 레이어: 바탕(원본 360) + 고른 효과 영상 섞기
  const COLORS = ["#ff465a", "#46beff", "#ffc832", "#6ee682", "#c882ff"];
  let presets = [], layers = [], pv = null;
  const KIND = {erp: "360 전체 (달라진 곳)", view: "사건 화면 (달라진 곳)", element: "효과만 (밝은 곳)"};

  function renderLayers() {
    $("elist").innerHTML = layers.map((l, k) => `
      <div class="elayer ${l.enabled ? "" : "off"}" data-id="${l.id}" style="--c:${COLORS[k % COLORS.length]}">
        <div><img src="${esc(l.thumb || "")}" alt=""><div class="muted" style="font-size:11px">${esc((l.size || []).join("×"))} · ${l.fps}fps · ${l.frames}f</div></div>
        <div>
          <div class="row" style="align-items:center">
            <div style="flex:0 0 auto"><label style="display:inline"><input type="checkbox" data-f="enabled" ${l.enabled ? "checked" : ""}> 사용</label></div>
            <div style="flex:1 1 160px"><input data-f="name" value="${esc(l.name || "")}"></div>
            <div style="flex:0 0 auto" class="muted">${esc(KIND[l.kind] || l.kind)}</div>
            <div style="flex:0 0 auto"><button data-del title="빼기">×</button></div>
          </div>
          <div class="row" style="align-items:flex-end">
            <div style="flex:0 0 90px"><label>시작 (초)</label><input data-f="start" type="number" step="0.1" min="0" value="${l.start}"></div>
            <div style="flex:0 0 90px"><label>세기</label><input data-f="opacity" type="number" step="0.05" min="0" max="1" value="${l.opacity}"></div>
            <div style="flex:0 0 130px"><label>섞는 방식</label><select data-f="blend">
              <option value="normal"${l.blend === "normal" ? " selected" : ""}>덮기</option>
              <option value="screen"${l.blend === "screen" ? " selected" : ""}>밝게 더하기</option></select></div>
            ${l.kind === "erp" ? "" : `
            <div style="flex:0 0 80px"><label>yaw</label><input data-f="yaw" type="number" step="1" value="${l.yaw}"></div>
            <div style="flex:0 0 80px"><label>pitch</label><input data-f="pitch" type="number" step="1" value="${l.pitch}"></div>
            <div style="flex:0 0 80px"><label>화각</label><input data-f="hfov" type="number" step="1" value="${l.hfov}"></div>`}
          </div>
        </div>
      </div>`).join("") || `<div class="muted" style="margin-top:8px">아직 효과 레이어가 없어요. 영상을 골라 '레이어 추가'를 눌러 주세요.</div>`;
    document.querySelectorAll("#elist .elayer").forEach(el => {
      el.querySelector("[data-del]").onclick = () => { collectLayers(); layers = layers.filter(l => l.id !== +el.dataset.id); renderLayers(); };
    });
  }

  function collectLayers() {
    document.querySelectorAll("#elist .elayer").forEach(el => {
      const l = layers.find(x => x.id === +el.dataset.id);
      if (!l) return;
      el.querySelectorAll("[data-f]").forEach(inp => {
        const k = inp.dataset.f;
        l[k] = inp.type === "checkbox" ? inp.checked : (inp.type === "number" ? Number(inp.value) : inp.value);
      });
    });
    return layers;
  }

  async function saveLayers() {
    collectLayers();
    const r = await post(`/api/run/${runId}/attempt/${n}/event/layers/update`, {layers});
    layers = r.layers; renderLayers();
  }

  function showPreview() {
    if (!pv) { $("epreview").classList.add("hidden"); return; }
    $("epreview").classList.remove("hidden");
    $("eslider").max = String(pv.frames.length - 1);
    const f = pv.frames[Math.min(+$("eslider").value, pv.frames.length - 1)];
    $("ecomp").src = f.comp; $("ematte").src = f.matte;
    $("ematte").classList.toggle("hidden", !$("ematteon").checked);
    $("etime").textContent = `${f.t}초 / ${pv.duration}초`;
    $("elegend").innerHTML = pv.layers.map((l, k) => {
      const c = f.cover[k];
      return `<span style="color:rgb(${l.color.join(",")})">■</span> ${esc(l.name || "")} — 이 시점 ${c == null ? "구간 밖" : Math.round(c * 1000) / 10 + "%"}` +
        ` · 최대 ${Math.round(l.peak_cover * 1000) / 10}% · 시작 ${Math.round(l.start_cover * 1000) / 10}% · 끝 ${Math.round(l.end_cover * 1000) / 10}%`;
    }).join("<br>") + `<br>매트 그림: 흐린 흑백 = 원본 그대로, 색 = 그 레이어가 섞이는 곳(진할수록 많이).`;
  }
  $("eslider").oninput = showPreview;
  $("ematteon").onchange = showPreview;

  $("epreset").onchange = () => {
    const p = presets.find(x => x.id === $("epreset").value);
    $("epinfo").textContent = p ? `${p.prompt}` : "I2V 영상(360 전체 또는 사건 화면)은 프리셋 없이 올리면 돼요.";
    $("epwf").href = p ? `/api/event_preset_workflow?id=${p.id}` : "#";
    $("epwf").style.visibility = p ? "visible" : "hidden"; $("epcopy").style.visibility = p ? "visible" : "hidden";
  };
  $("epcopy").onclick = async () => {
    const p = presets.find(x => x.id === $("epreset").value);
    if (!p) return;
    try { await navigator.clipboard.writeText(p.prompt); $("ecstate").textContent = "효과 프롬프트를 복사했어요."; } catch { }
  };

  $("eadd").onclick = async () => {
    const f = $("efile").files[0];
    if (!f) { $("ecstate").textContent = "ComfyUI 에서 만든 MP4 를 골라 주세요."; return; }
    $("eadd").disabled = true; $("ecstate").textContent = "올리는 중…";
    try {
      collectLayers();
      const fd = new FormData(); fd.append("video", f); fd.append("preset", $("epreset").value || "");
      const r = await api(`/api/run/${runId}/attempt/${n}/event/layers`, {method: "POST", body: fd});
      layers = r.layers; renderLayers(); $("efile").value = "";
      $("ecstate").textContent = "레이어를 추가했어요. '매트 확인'으로 섞이는 곳을 확인하세요.";
    } catch (e) { $("ecstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("eadd").disabled = false;
  };
  $("elsave").onclick = async () => {
    try { await saveLayers(); $("ecstate").textContent = "레이어를 저장했어요."; }
    catch (e) { $("ecstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
  $("eprev").onclick = async () => {
    $("eprev").disabled = true; $("ecstate").textContent = "레이어를 읽고 매트를 계산하는 중… (처음엔 10~60초)";
    try {
      await saveLayers();
      pv = await post(`/api/run/${runId}/attempt/${n}/event/preview`, {count: 8});
      $("eslider").value = String(Math.floor(pv.frames.length / 2));
      $("ecstate").textContent = `미리보기 ${pv.frames.length}장 · 전체 ${pv.duration}초`;
      showPreview();
    } catch (e) { $("ecstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("eprev").disabled = false;
  };
  $("ecompose").onclick = async () => {
    $("ecompose").disabled = true; $("ecstate").textContent = "레이어를 360 에 합치는 중… (4K 5초 기준 1~3분)";
    try {
      await saveLayers();
      const r = await post(`/api/run/${runId}/attempt/${n}/event/compose_layers`, {});
      ev = {...(ev || {}), result: r};
      $("ecstate").textContent = `완료 · ${r.finished || ""}`;
      showResult(r);
    } catch (e) { $("ecstate").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("ecompose").disabled = false;
  };

  $("equest").onclick = async () => {
    try { const r = await post(`/api/run/${runId}/attempt/${n}/event/to_viewer`, {});
          $("einfo").textContent = `Quest 뷰어에 8K 사진 + 사건 영상을 함께 추가했어요 (목록 ${r.count}개). 뷰어에서 사진 → 영상 → 사진으로 재생돼요 — Unity 에서 다시 빌드하세요`; }
    catch (e) { $("einfo").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };

  return {
    show(id, att, cfg) {
      if (!att || att.state !== "done" || !(att.files || []).length) { this.hide(); return; }
      runId = id; n = att.n; ev = att.event || null;
      if (cfg && cfg.event_presets && !presets.length) {
        presets = cfg.event_presets;
        $("epreset").innerHTML = `<option value="">— 프리셋 없음 (I2V 영상) —</option>` +
          presets.map(p => `<option value="${esc(p.id)}">${esc(p.name_ko)} (${p.blend === "screen" ? "밝게 더하기" : "덮기"})</option>`).join("");
      }
      $("epreset").onchange();
      layers = (ev && ev.layers) ? ev.layers.slice() : [];
      pv = null; renderLayers(); showPreview();
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
