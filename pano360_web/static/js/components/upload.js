// 1단계: 사진 올리기 · 작업자 메모 · LLM 분석 요청
import {api, post} from "../core/api.js";
import {$, errHtml, usageText, fillModels, show} from "../core/dom.js";
import {store} from "../core/store.js";
import {afterUpload} from "../core/actions.js";
import {requestButton, createGate} from "../core/request.js";

export function setupUpload() {
  const sendUpload = createGate($("upstate"));

  function upload(file) {
    return sendUpload(async () => {
      show("srcbox");
      $("srcimg").src = URL.createObjectURL(file);
      $("srcinfo").textContent = "업로드 중…";
      const fd = new FormData(); fd.append("image", file);
      try {
        const up = await api("/api/upload", {method: "POST", body: fd});
        await afterUpload(up);
        $("srcinfo").innerHTML = `${up.width} × ${up.height}px` +
          (up.is_erp ? " · 360 이미지" : (up.exif_hfov ? ` · EXIF 화각 ${up.exif_hfov}°` : " · EXIF 화각 없음 (LLM이 추정)"));
      } catch (e) { $("srcinfo").innerHTML = errHtml(e.message); throw e; }
    });
  }

  $("drop").onclick = () => $("file").click();
  $("file").onchange = e => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; };
  ["dragenter", "dragover"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
  $("drop").addEventListener("drop", e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

  // 분석 — 실패해도 이 버튼만 다시 누르면 됨
  requestButton($("analyze"), async () => {
    const id = store.runId();
    if (!id) { $("anstate").innerHTML = errHtml("먼저 사진을 올리세요."); return false; }
    $("anstate").textContent = "분석 중… (10~40초)";
    try {
      const r = await post("/api/analyze", {run_id: id, note: $("note").value, model: $("model").value});
      if (!store.isRun(id)) return;
      store.emit("analysis", {analysis: r.analysis, placement: r.placement, prompts: r.prompts});
      $("anstate").textContent = usageText(r.model, r.usage);
    } catch (e) {
      if (store.isRun(id)) $("anstate").innerHTML = errHtml(e.message);
      throw e;
    }
  });

  store.on("cfg", cfg => cfg && fillModels($("model"), cfg));
  store.on("run", m => {
    $("anstate").textContent = "";
    if (!m) return;
    show("srcbox");
    $("srcimg").src = `/runs/${m.run_id}/source${m.source_ext}`;
    $("srcinfo").textContent = `${m.width} × ${m.height}px · ${m.original_name || ""}`;
    $("note").value = m.note || "";
    if (m.model) $("model").value = m.model;
    show("erpbox", !!m.is_erp);
  });
}
