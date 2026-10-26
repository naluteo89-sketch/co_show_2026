// 최근 작업 목록 (옆 칸) — 누르면 그 작업을 엶
import {api, esc} from "../core/api.js";
import {$} from "../core/dom.js";
import {store} from "../core/store.js";
import {openRun} from "../core/actions.js";

export function setupRuns() {
  async function load() {
    try {
      const rs = await api("/api/runs");
      $("runs").innerHTML = rs.length ? rs.map(r => `
        <div class="r" data-id="${esc(r.run_id)}">
          <img src="${esc(r.last && r.last.files[0] ? r.last.files[0].url : r.thumb)}" alt="">
          <div class="t"><div>${esc(r.name || r.run_id)}</div><div class="muted">${esc(r.run_id.slice(0, 15))} · ${r.attempts}회</div></div>
        </div>`).join("") : "아직 없어요";
      $("runs").querySelectorAll(".r").forEach(el => el.onclick = () =>
        openRun(el.dataset.id).catch(e => { $("runs").insertAdjacentHTML("afterbegin", `<div class="err">${esc(e.message)}</div>`); }));
    } catch (e) { $("runs").textContent = "목록을 불러오지 못했어요"; }
  }
  store.on("runs:changed", load);
  store.on("attempts", load);
  load();
}
