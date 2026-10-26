// 머리글: 연결 상태 칩 (설정을 받으면 그림)
import {esc} from "../core/api.js";
import {$} from "../core/dom.js";
import {store} from "../core/store.js";

export function setupHeader() {
  const chip = (ok, t) => `<span class="chip ${ok ? "ok" : "bad"}">${esc(t)}</span>`;
  store.on("cfg", cfg => {
    if (!cfg) { $("chips").innerHTML = `<span class="chip bad">서버 연결 실패</span>`; return; }
    $("chips").innerHTML = chip(cfg.openai_ready, `OpenAI · 기본 ${cfg.openai_model}`) +
      cfg.backends.map(b => chip(b.ok, b.label)).join("") +
      `<span class="chip">${esc(cfg.workflow)}</span>` +
      `<span class="chip" title="${esc(cfg.version)}">v ${esc((cfg.version || "?").split(" ").slice(0, 2).join(" "))}</span>`;
  });
}
