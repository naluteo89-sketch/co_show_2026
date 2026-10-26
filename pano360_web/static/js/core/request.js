// 요청 버튼: 누르면 요청을 보내고, 보냈는지 버튼 옆에 표시하고, 최소 5초 동안 다시 누를 수 없게 한다.
//
//   requestButton(button, async () => { ...서버 요청... })
//
// 표시:  보내는 중…  →  ✓ 보냄 14:03:21 (서버 받음)   또는   ✗ 보내지 못함: 원인
// handler 가 false 를 돌려주면 보내지 않은 것(입력 확인 등)으로 보고 바로 풀어 준다.

export const COOLDOWN_MS = 5000;

const pad = n => String(n).padStart(2, "0");
const clock = d => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

// 버튼 옆 상태 칸 (버튼이 <a> 안에 있으면 그 뒤에)
export function stateFor(button) {
  if (button._reqState) return button._reqState;
  const host = button.parentElement && button.parentElement.tagName === "A" ? button.parentElement : button;
  let el = host.nextElementSibling;
  if (!el || !el.classList.contains("reqstate")) {
    el = document.createElement("span");
    el.className = "reqstate";
    el.setAttribute("aria-live", "polite");
    host.insertAdjacentElement("afterend", el);
  }
  button._reqState = el;
  return el;
}

export function setRequestState(el, kind, text) {
  el.className = `reqstate ${kind}`;
  el.textContent = text;
}

// 버튼 없이 보내는 요청(끌어다 놓기 업로드 등)도 같은 규칙으로
export function createGate(stateEl, cooldown = COOLDOWN_MS) {
  let until = 0, busy = false;
  return async function send(handler) {
    const now = Date.now();
    if (busy || now < until) {
      setRequestState(stateEl, "wait", `잠시 후 다시 (${Math.ceil(Math.max(until - now, 0) / 1000) || 1}초)`);
      return;
    }
    busy = true; until = now + cooldown;
    setRequestState(stateEl, "sending", "보내는 중…");
    try {
      const r = await handler();
      if (r === false) { until = 0; setRequestState(stateEl, "", ""); }
      else setRequestState(stateEl, "ok", `✓ 보냄 ${clock(new Date())} (서버 받음)`);
    } catch (e) {
      setRequestState(stateEl, "fail", `✗ 보내지 못함: ${e.message || e}`);
    } finally { busy = false; }
  };
}

export function requestButton(button, handler, {cooldown = COOLDOWN_MS} = {}) {
  if (!button) return;
  const label = button.dataset.label || button.textContent;
  button.dataset.label = label;
  const state = stateFor(button);
  let locked = false;

  async function onClick(e) {
    e.preventDefault();
    if (locked) return;
    locked = true;
    button.disabled = true;
    const t0 = Date.now();
    setRequestState(state, "sending", "보내는 중…");
    let cancelled = false;
    try {
      const r = await handler(e);
      if (r === false) { cancelled = true; setRequestState(state, "", ""); }
      else setRequestState(state, "ok", `✓ 보냄 ${clock(new Date())} (서버 받음)`);
    } catch (err) {
      setRequestState(state, "fail", `✗ 보내지 못함: ${err.message || err}`);
    }
    if (!cancelled) {
      // 쿨타임: 요청이 빨리 끝나도 누른 때부터 최소 cooldown 동안 잠금
      while (Date.now() - t0 < cooldown) {
        button.textContent = `${label} (${Math.ceil((cooldown - (Date.now() - t0)) / 1000)})`;
        await new Promise(r => setTimeout(r, 250));
      }
    }
    button.textContent = label;
    button.disabled = false;
    locked = false;
  }
  button.addEventListener("click", onClick);
}
