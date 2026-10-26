// 6단계 안: 움직일 범위(마스크) 칠하기 · 저장 · PNG 받기
// 다른 컴포넌트(영상 만들기)는 store.services.mask.flush() 로 '칠했는데 저장 안 한 범위'만 저장해 달라고 한다.
import {api, post} from "../core/api.js";
import {$, errHtml} from "../core/dom.js";
import {store} from "../core/store.js";
import {requestButton} from "../core/request.js";

export function setupEventMask() {
  let runId = null, n = null;
  // ---- 움직일 범위 칠하기 (360 그림 위, 흰색 = 영상)
  const cv = $("emask"), cx = cv.getContext("2d");
  let erasing = false, painting = false, maskDirty = false, last = null;
  function paintAt(e) {
    const b = cv.getBoundingClientRect();
    const x = (e.clientX - b.left) / b.width * cv.width, y = (e.clientY - b.top) / b.height * cv.height;
    const r = +$("ebrush").value / 2;
    cx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
    cx.strokeStyle = cx.fillStyle = "rgb(255,40,70)"; cx.lineWidth = r * 2; cx.lineCap = "round";
    if (last) { cx.beginPath(); cx.moveTo(last[0], last[1]); cx.lineTo(x, y); cx.stroke();
      // 360 좌우가 이어지므로 가장자리에서 칠하면 반대쪽에도
      if (x < r || x > cv.width - r) { const sh = x < r ? cv.width : -cv.width; cx.beginPath(); cx.moveTo(last[0] + sh, last[1]); cx.lineTo(x + sh, y); cx.stroke(); } }
    else { cx.beginPath(); cx.arc(x, y, r, 0, Math.PI * 2); cx.fill(); }
    last = [x, y]; maskDirty = true; $("emstate").textContent = "칠하는 중 — '범위 저장'을 눌러 주세요";
  }
  $("epaint").addEventListener("pointerdown", e => { if (e.button !== 0) return; e.preventDefault();
    $("epaint").setPointerCapture(e.pointerId); painting = true; last = null; paintAt(e); });
  $("epaint").addEventListener("pointermove", e => { if (painting) paintAt(e); });
  ["pointerup", "pointercancel"].forEach(t => $("epaint").addEventListener(t, () => { painting = false; last = null; }));
  $("eerase").onclick = () => { erasing = !erasing; $("eerase").textContent = erasing ? "칠하기" : "지우개"; };
  $("eclear").onclick = () => { cx.clearRect(0, 0, cv.width, cv.height); maskDirty = true; $("emstate").textContent = "지웠어요 — '범위 저장'을 눌러 주세요"; };
  function maskPng() {                                       // 칠한 곳(빨강) → 흰색 PNG
    const d = cx.getImageData(0, 0, cv.width, cv.height), t = document.createElement("canvas");
    t.width = cv.width; t.height = cv.height;
    const tx = t.getContext("2d"), o = tx.createImageData(cv.width, cv.height);
    let any = 0;
    for (let i = 0; i < d.data.length; i += 4) { const a = d.data[i + 3]; any += a > 0;
      o.data[i] = o.data[i + 1] = o.data[i + 2] = a; o.data[i + 3] = 255; }
    tx.putImageData(o, 0, 0);
    return any ? t.toDataURL("image/png") : "";
  }
  async function saveMask() {
    const r = await post(`/api/run/${runId}/attempt/${n}/event/mask`, {png: maskPng()});
    maskDirty = false;
    $("emstate").textContent = r.mask ? `범위 저장 — 360 의 ${Math.round(r.cover * 1000) / 10}%` : "범위를 지웠어요";
    setMaskLink(r.mask);
    return r;
  }
  requestButton($("emsave"), async () => {
    if (!runId) return false;
    try { await saveMask(); } catch (e) { $("emstate").innerHTML = errHtml(e.message); throw e; }
  });
  function setMaskLink(url) {
    $("emdl").classList.toggle("hidden", !url);
    if (url) { $("emdl").href = url; $("emdl").setAttribute("download", `${runId}_a${String(n).padStart(2, "0")}_mask.png`); }
  }
  async function loadBase() {
    cx.clearRect(0, 0, cv.width, cv.height); maskDirty = false;
    try {
      const id = runId, k = n;
      const b = await api(`/api/run/${id}/attempt/${k}/event/base`);
      if (id !== runId || k !== n) return;
      $("embase").src = b.base;
      setMaskLink(b.mask);
      if (b.mask) {
        const im = new Image();
        im.onload = () => {                                    // 흰색 마스크 → 빨강 칠
          const t = document.createElement("canvas"); t.width = cv.width; t.height = cv.height;
          const tx = t.getContext("2d"); tx.drawImage(im, 0, 0, cv.width, cv.height);
          const d = tx.getImageData(0, 0, cv.width, cv.height), o = cx.createImageData(cv.width, cv.height);
          for (let i = 0; i < d.data.length; i += 4) { o.data[i] = 255; o.data[i + 1] = 40; o.data[i + 2] = 70; o.data[i + 3] = d.data[i]; }
          cx.putImageData(o, 0, 0);
        };
        im.src = b.mask;
        $("emstate").textContent = "저장된 범위를 불러왔어요";
      } else $("emstate").textContent = "움직일 곳을 칠하고 '범위 저장'을 누르세요";
    } catch (e) { $("emstate").innerHTML = errHtml(e.message); }
  }


  store.services.mask = { flush: async () => { if (maskDirty && runId) await saveMask(); } };

  store.on("target", att => {
    if (!att) { runId = n = null; return; }
    runId = store.runId(); n = att.n;
    loadBase();
  });
}
