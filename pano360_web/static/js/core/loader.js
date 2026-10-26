// HTML 조각 불러오기: <div data-partial="upload"></div> → /static/html/upload.html 내용으로 바꿈
// 조각 안에 또 조각이 있으면 그것도 차례로 불러온다 (최대 5단계).
async function loadOne(slot) {
  const name = slot.dataset.partial;
  try {
    const r = await fetch(`/static/html/${name}.html`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const tpl = document.createElement("template");
    tpl.innerHTML = await r.text();
    slot.replaceWith(tpl.content);
  } catch (e) {
    slot.removeAttribute("data-partial");
    slot.className = "card err";
    slot.textContent = `화면 조각을 불러오지 못했어요: ${name}.html (${e.message})`;
  }
}

export async function loadPartials(root = document) {
  for (let depth = 0; depth < 5; depth++) {
    const slots = [...root.querySelectorAll("[data-partial]")];
    if (!slots.length) return;
    await Promise.all(slots.map(loadOne));
  }
}
