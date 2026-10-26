// 6단계 안: 360 전체 영상 프롬프트 (LTX + 360 LoRA 용) — 고쳐서 복사
// 사건을 만들거나 고치면 그 내용으로 채워지고, 없으면 기본 프롬프트.
import {$, copyText} from "../core/dom.js";
import {store} from "../core/store.js";

export function setupEventPrompt() {
  const fill = lp => { $("eltxpos").value = lp?.positive || ""; $("eltxneg").value = lp?.negative || ""; };
  const copyTo = (txt, what) => copyText(txt, $("eltxstate"), `${what} 를 복사했어요.`);
  $("eltxcp").onclick = () => copyTo($("eltxpos").value, "Positive");
  $("eltxcn").onclick = () => copyTo($("eltxneg").value, "Negative");

  store.on("target", att => {
    if (!att) return;
    fill((att.event && att.event.ltx360) || store.cfg?.ltx360);
    $("eltxstate").textContent = "";
  });
  store.on("event", ev => { if (ev && ev.ltx360) fill(ev.ltx360); });
}
