// 화면 공통 도우미 (app.js · video.js · event.js 가 함께 씀)
import {esc} from "./api.js";

export const $ = id => document.getElementById(id);
export const errHtml = msg => `<span class="err">${esc(msg)}</span>`;
export const usageText = (model, u) => `${model} · 입력 ${u?.input_tokens ?? "?"} / 출력 ${u?.output_tokens ?? "?"} 토큰`;
export const modelOptions = models => models.map(m => `<option>${esc(m)}</option>`).join("");

// 실행 위치 목록을 select 에 채우고, 기본(연결된) 실행 위치를 골라 돌려줌
export function fillBackends(select, cfg) {
  select.innerHTML = cfg.backends.map(b =>
    `<option value="${esc(b.name)}">${esc(b.label)}${b.ok ? "" : " (연결 안 됨)"}</option>`).join("");
  const pref = cfg.backends.find(b => b.name === cfg.default_backend && b.ok) || cfg.backends.find(b => b.ok);
  if (pref) select.value = pref.name;
  return pref;
}

export async function copyText(text, stateEl, okMsg, failMsg = "복사하지 못했어요. 칸에서 직접 복사하세요.") {
  try { await navigator.clipboard.writeText(text); stateEl.textContent = okMsg; }
  catch { stateEl.textContent = failMsg; }
}
