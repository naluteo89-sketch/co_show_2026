// 5·6단계 '대상 결과' 고르기 — 완료된 결과 중 하나. 두 칸(5·6)의 선택은 같이 움직인다.
// 새 시도가 실패하거나 진행 중이어도 이 선택은 바뀌지 않는다.
import {esc} from "../core/api.js";
import {store, doneAttempts} from "../core/store.js";
import {setTarget} from "../core/actions.js";

const label = a => `#${a.n} · ${a.mode === "import" ? "가져온 360" : a.mode === "enhance" ? "8K 화질 개선" : a.mode === "pole" ? "발밑·하늘 보정" : "생성"} · ${a.created || ""}`;

export function setupTargetSelect() {
  const selects = () => [...document.querySelectorAll("[data-target-select]")];

  function render() {
    const done = doneAttempts(store.attempts).slice().reverse();
    const html = done.length ? done.map(a => `<option value="${a.n}">${esc(label(a))}</option>`).join("")
                             : `<option value="">완료된 결과 없음</option>`;
    selects().forEach(sel => { sel.innerHTML = html; sel.value = store.target ? String(store.target.n) : ""; });
  }

  selects().forEach(sel => sel.addEventListener("change", () => {
    const a = doneAttempts(store.attempts).find(x => x.n === +sel.value);
    if (a) setTarget(a, true);
  }));
  store.on("pick-target", n => {
    const a = doneAttempts(store.attempts).find(x => x.n === n);
    if (a) { setTarget(a, true); render(); document.getElementById("motioncard")?.scrollIntoView({behavior: "smooth"}); }
  });
  store.on("attempts", render);
  store.on("target", render);
}
