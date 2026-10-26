// 시작: HTML 조각을 붙이고 → 컴포넌트를 연결하고 → 서버 설정을 받는다.
//
// 구조
//   html/            단계별 마크업 조각 (components/ 의 같은 이름 파일과 짝)
//   css/             base · layout · components/
//   js/core/         api · dom · store(상태·알림) · actions(작업 열기·생성 요청) · request(쿨타임·보냄 표시) · poller · loader
//   js/components/   단계별 동작. 서로 직접 부르지 않고 store 알림으로만 주고받음
import {api} from "./core/api.js";
import {store} from "./core/store.js";
import {loadPartials} from "./core/loader.js";
import {setupHeader} from "./components/header.js";
import {setupUpload} from "./components/upload.js";
import {setupErpTools} from "./components/erp-tools.js";
import {setupAnalysis} from "./components/analysis.js";
import {setupPrompts} from "./components/prompts.js";
import {setupResult} from "./components/result.js";
import {setupAttempts} from "./components/attempts.js";
import {setupTargetSelect} from "./components/target-select.js";
import {setupMotion} from "./components/motion.js";
import {setupEventPlan} from "./components/event-plan.js";
import {setupEventMask} from "./components/event-mask.js";
import {setupEventPrompt} from "./components/event-prompt.js";
import {setupEventVideo} from "./components/event-video.js";
import {setupRuns} from "./components/runs.js";

const components = [setupHeader, setupUpload, setupErpTools, setupAnalysis, setupPrompts, setupResult, setupAttempts,
                    setupTargetSelect, setupMotion, setupEventPlan, setupEventMask, setupEventPrompt, setupEventVideo, setupRuns];

async function start() {
  await loadPartials();
  for (const setup of components) {
    try { setup(); } catch (e) { console.error(`[studio] ${setup.name} 시작 실패`, e); }   // 한 컴포넌트가 실패해도 나머지는 동작
  }
  try { store.cfg = await api("/api/config"); }
  catch (e) { store.cfg = null; }
  store.emit("cfg", store.cfg);
}

start();
