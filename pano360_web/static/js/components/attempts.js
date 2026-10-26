// 4단계 안: 지금까지의 시도 목록 (360 보기 · 이 설정 다시 불러오기)
import {esc} from "../core/api.js";
import {$, finalFile} from "../core/dom.js";
import {store} from "../core/store.js";

const fmtSet = s => s && Object.keys(s).length
  ? ` · steps ${s.steps ?? "-"} · CFG ${s.cfg ?? "-"} · LoRA ${s.lora_strength ?? "-"} · seam ${s.seam_denoise ?? "-"}` : "";

function head(a) {
  if (a.mode === "import") return `#${a.n} · 가져온 360 이미지 (생성 없음) · ${a.created}`;
  if (a.mode === "enhance" || a.mode === "pole")
    return `#${a.n} · ${a.backend} · ${a.state} · ${a.mode === "pole" ? `발밑·하늘 보정${a.from_attempt ? ` (#${a.from_attempt} 결과)` : ""} + 8K` : "8K 화질 개선"} · ${a.created}`;
  return `#${a.n} · ${a.backend} · ${a.state} · seed ${a.seeds?.seed1}/${a.seeds?.seed2}${fmtSet(a.settings)} · ${a.created}`;
}

export function setupAttempts() {
  store.on("attempts", attempts => {
    const m = store.run;
    $("attempts").innerHTML = attempts.slice().reverse().map(a => {
      const plain = a.mode === "enhance" || a.mode === "pole" || a.mode === "import";
      const f = finalFile(a.files);
      return `<details class="attempt" data-n="${a.n}"><summary>${esc(head(a))}</summary>
        ${f ? `<a href="#" data-url="${esc(f.jpg || f.url)}" class="show">360 보기</a>` : ""}
        ${f ? `<a href="#" class="totarget">5·6단계 대상으로</a>` : ""}
        ${plain || !a.prompts ? "" : `<a href="#" class="reuse">이 설정 다시 불러오기</a>
        <pre>${esc(a.prompts.positive)}\n\nNEG: ${esc(a.prompts.negative)}\n\nyaw ${a.placement?.yaw} · pitch ${a.placement?.pitch} · hFOV ${a.placement?.hfov}</pre>`}
        ${a.error ? `<div class="err">${esc(a.error)}</div>` : ""}
      </details>`;
    }).join("");
    $("attempts").querySelectorAll(".attempt").forEach(el => {
      const a = attempts.find(x => x.n === +el.dataset.n);
      el.querySelector(".show")?.addEventListener("click", e => { e.preventDefault(); store.emit("view", e.target.dataset.url); });
      el.querySelector(".totarget")?.addEventListener("click", e => { e.preventDefault(); store.emit("pick-target", a.n); });
      el.querySelector(".reuse")?.addEventListener("click", e => {
        e.preventDefault();
        store.emit("analysis", {analysis: m.analysis || {}, placement: a.placement, prompts: a.prompts});
        store.emit("reuse", a);
      });
    });
  });
  store.on("run", m => { if (!m) $("attempts").innerHTML = ""; });
}
