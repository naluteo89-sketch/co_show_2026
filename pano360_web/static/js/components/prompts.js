// 3단계: 프롬프트 · 생성 설정 · 360 이미지 생성 요청 · 모델 점검
import {api, esc} from "../core/api.js";
import {$, show, fillBackends} from "../core/dom.js";
import {store} from "../core/store.js";
import {submitJob} from "../core/actions.js";
import {requestButton} from "../core/request.js";

const numberValue = id => $(id).value.trim() === "" ? null : Number($(id).value);
const linkify = s => esc(s).replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

export function setupPrompts() {
  const settings = () => ({steps: numberValue("g_steps"), cfg: numberValue("g_cfg"),
    lora_strength: numberValue("g_lora"), seam_denoise: numberValue("g_seam"),
    version: $("g_version").value, viewer_scale: numberValue("g_viewer")});
  store.services.settings = settings;

  store.on("analysis", ({analysis, prompts: pr}) => {
    show("promptcard");
    $("p_pos").value = pr.positive || ""; $("p_neg").value = pr.negative || "";
    $("p_spos").value = pr.seam_positive || ""; $("p_sneg").value = pr.seam_negative || "";
    $("p_bpos").value = pr.back_positive || "";
    $("raw").textContent = JSON.stringify(analysis, null, 2);
  });
  store.on("run", m => { if (!m) show("promptcard", false); });

  // '이 설정 다시 불러오기' (결과 칸의 시도 목록에서)
  store.on("reuse", a => {
    if (a.seeds) { $("seed1").value = a.seeds.seed1; $("seed2").value = a.seeds.seed2; }
    const s = a.settings || {};
    if (s.steps) $("g_steps").value = s.steps;
    if (s.cfg) $("g_cfg").value = s.cfg;
    if (s.lora_strength != null) $("g_lora").value = s.lora_strength;
    if (s.seam_denoise) $("g_seam").value = s.seam_denoise;
    if (s.version) $("g_version").value = s.version;
    if (s.viewer_scale) $("g_viewer").value = s.viewer_scale;
    $("promptcard").scrollIntoView({behavior: "smooth"});
  });

  store.on("cfg", cfg => {
    if (!cfg) return;
    fillBackends($("backend"), cfg);
    const g = cfg.gen_defaults || {};
    $("g_steps").value = g.steps ?? ""; $("g_cfg").value = g.cfg ?? "";
    $("g_lora").value = g.lora_strength ?? ""; $("g_seam").value = g.seam_denoise ?? "";
    $("g_version").value = g.version || "A"; $("g_viewer").value = g.viewer_scale ?? 0.6;
  });

  requestButton($("gen"), async () => {
    if (!$("p_pos").value.startsWith("Fill the green spaces") &&
        !confirm("Positive 프롬프트가 LoRA 트리거 문장으로 시작하지 않아요. 그대로 진행할까요?")) return false;
    return submitJob({backend: $("backend").value, placement: store.services.placement(),
      seed1: numberValue("seed1"), seed2: numberValue("seed2"), settings: settings(),
      prompts: {positive: $("p_pos").value, negative: $("p_neg").value,
        seam_positive: $("p_spos").value, seam_negative: $("p_sneg").value, back_positive: $("p_bpos").value}},
      r => `대기 중… (seed ${r.seeds.seed1} / ${r.seeds.seed2})`);
  });

  requestButton($("mcheck"), async () => {
    const box = $("mcheckout"); show("mcheckout");
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
    } catch (e) { box.innerHTML = `<div class="bad">${esc(e.message)}</div>`; throw e; }
  });
}
