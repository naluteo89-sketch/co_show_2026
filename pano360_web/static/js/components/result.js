// 4단계: 진행 상태 · 결과 360 (미리보기 · 다운로드 · 뷰어로 보내기 · 원본 자리 화질 · 발밑·하늘 보정)
// 새 작업이 진행 중이어도 이전 결과는 그대로 보여 주고, 실패해도 다른 버튼은 막지 않는다.
import {post} from "../core/api.js";
import {$, show, errHtml, finalFile, withV} from "../core/dom.js";
import {store} from "../core/store.js";
import {refreshRun, submitJob} from "../core/actions.js";
import {createPoller} from "../core/poller.js";
import {requestButton} from "../core/request.js";
import {createViewer} from "./viewer.js";

export function setupResult() {
  const viewer = createViewer($("viewer"));
  let shownN = null;

  const poller = createPoller(
    async (id, n, signal) => {               // 실패한 작업도 상태(state: "error")로 받음 — 오류 응답과 구분
      const r = await fetch(`/api/run/${id}/attempt/${n}`, {signal});
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.state) throw new Error(j.error || `HTTP ${r.status}`);
      return j;
    },
    onStatus,
    (error, wait) => { $("jobtext").textContent = `상태 확인 실패: ${error.message} — ${Math.round(wait / 1000)}초 뒤 다시 확인해요`; }
  );

  function onStatus(s, n) {
    if (s.state === "queued") $("jobtext").textContent = `#${n} 대기열에서 기다리는 중…`;
    else if (s.state === "running") {
      $("jobtext").textContent = `#${n} ${s.node_title || "실행"} 중… ${s.max ? `${s.value}/${s.max}` : ""}`;
      $("bar").style.width = (s.max ? Math.round(s.value / s.max * 100) : 5) + "%";
    } else if (s.state === "done") {
      $("bar").style.width = "100%";
      $("jobtext").textContent = `#${n} 완료 · ${s.finished || ""}`;
      refreshRun().then(() => showResult(store.attempts.find(a => a.n === n) || s)).catch(() => showResult(s));
    } else if (s.state === "error") {
      $("jobtext").innerHTML = errHtml(`#${n} 실패: ${s.error}`) + ' <span class="muted">— 설정을 고쳐 같은 버튼으로 다시 보내면 돼요</span>';
      refreshRun().catch(() => {});
    }
  }

  function showResult(att) {
    if (!att) return;
    const f = finalFile(att.files);
    if (!f) return;
    shownN = att.n;
    show("jobcard");
    const url = (f.jpg || f.url) + withV(f);
    $("results").innerHTML = `<div>
      <div class="row resultbtns">
        <a href="${url}" download><button class="primary">360 이미지 다운로드 (JPG)</button></a>
        <a href="${f.url + withV(f)}" download><button>원본 PNG</button></a>
        ${f.jpg ? `<button class="toviewer">Quest 뷰어에 추가</button>` : ""}
      </div>
      <a href="${url}" target="_blank"><img src="${url}" alt="" class="pano"></a>
      <div class="muted dims"></div></div>`;
    const im = $("results").querySelector("img.pano");
    im.onload = () => { $("results").querySelector(".dims").textContent =
      `#${att.n} · 360 결과 ${im.naturalWidth} × ${im.naturalHeight}px · 위 360 뷰어 화면은 미리보기라 저장해도 작은 화면 크기로만 저장돼요`; };

    if (att.mode !== "pole" && att.mode !== "enhance" && att.mode !== "import" && !String(att.photo_pasted || "").startsWith("upscaled"))
      $("results").insertAdjacentHTML("beforeend", `<div class="note">원본 사진 자리가 주변보다 흐리면 →
        <button class="repaste">원본 자리 화질 맞추기</button>
        <span class="muted small">원본 사진을 원래 해상도로 AI 업스케일해서 같은 자리에 다시 붙여요 (1~2분, 이 결과를 고침)</span></div>`);
    $("results").insertAdjacentHTML("beforeend", `<div class="note">발밑이 뭉개지거나 하늘 꼭대기가 비어 보이면 →
      <button class="pole">발밑·하늘 보정 + 8K</button>
      <span class="muted small">극점 부분만 다시 그려서 합성한 뒤 8K로 다시 만들어요 (새 시도로 저장)</span></div>`);

    const id = store.runId(), n = att.n;
    requestButton($("results").querySelector(".toviewer"), async () => {
      const r = await post(`/api/run/${id}/attempt/${n}/to_viewer`, {});
      $("jobtext").textContent = `#${n} Quest 뷰어에 추가됨 (사진 ${r.count}장) — Unity에서 다시 빌드하세요`;
    });
    requestButton($("results").querySelector(".repaste"), async () => {
      $("jobtext").textContent = `#${n} 원본 사진 업스케일 중…`;
      const r = await post(`/api/run/${id}/attempt/${n}/repaste`, {});
      $("jobtext").textContent = `#${n} 원본 자리 화질 맞추기 완료 (${r.mode})`;
      if (store.isRun(id)) showResult({...att, files: r.files, photo_pasted: r.mode});
    });
    requestButton($("results").querySelector(".pole"), () =>
      submitJob({backend: $("backend").value || $("backend2").value, mode: "pole", from_attempt: n,
                 settings: store.services.settings?.()}, () => "발밑·하늘 보정 대기 중…"));
    viewer.load(url);
  }

  store.on("job", ({runId, n, message}) => {
    if (!store.isRun(runId)) return;
    show("jobcard");
    $("bar").style.width = "0%";
    $("jobtext").textContent = `#${n} ${message}`;
    poller.start(runId, n);
    $("jobcard").scrollIntoView({behavior: "smooth", block: "nearest"});
  });
  store.on("show-result", showResult);
  store.on("view", url => viewer.load(url));
  store.on("run", m => {
    poller.stop();
    shownN = null;
    $("results").innerHTML = ""; $("jobtext").textContent = ""; $("bar").style.width = "0";
    viewer.hide();
    if (!m || !(m.attempts || []).length) { show("jobcard", false); return; }
    show("jobcard");
    const done = m.attempts.slice().reverse().find(a => (a.files || []).length);
    if (done) showResult(done);
  });
}
