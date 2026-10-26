// 여러 컴포넌트가 함께 쓰는 동작 (작업 열기 · 새로고침 · 생성 요청 · 5·6단계 대상 고르기)
import {api, post} from "./api.js";
import {store, doneAttempts} from "./store.js";

let loadVersion = 0;

// 작업 열기: 화면을 그 작업으로 바꿈
export async function openRun(id) {
  const version = ++loadVersion;
  store.run = null; store.attempts = []; store.target = null; store.targetPinned = false;
  store.emit("run", null);
  store.emit("target", null);
  const m = await api(`/api/run/${id}`);
  if (version !== loadVersion) return null;
  store.run = m;
  store.emit("run", m);
  applyAttempts(m.attempts || []);
  const pending = (m.attempts || []).find(a => a.state !== "done" && a.state !== "error");
  if (pending) store.emit("job", {runId: id, n: pending.n, message: "진행 중인 작업을 이어서 확인해요…"});
  return m;
}

// 같은 작업의 시도 목록만 다시 읽음 (화면의 입력값은 건드리지 않음)
export async function refreshRun() {
  const id = store.runId();
  if (!id) return null;
  const m = await api(`/api/run/${id}`);
  if (!store.isRun(id)) return null;
  store.run = m;
  applyAttempts(m.attempts || []);
  return m;
}

function applyAttempts(attempts) {
  store.attempts = attempts;
  store.emit("attempts", attempts);
  const done = doneAttempts(attempts);
  const cur = store.target && done.find(a => a.n === store.target.n);
  if (cur) { store.target = cur; return; }               // 같은 결과면 조용히 갱신 (5·6단계 입력 유지)
  const next = store.targetPinned ? done.find(a => a.n === store.target?.n) : done[done.length - 1];
  setTarget(next || null, false);
}

// 5·6단계 대상 결과 바꾸기 (같은 결과면 알리지 않음)
export function setTarget(att, pinned = true) {
  if (pinned) store.targetPinned = !!att;
  const same = store.target && att && store.target.n === att.n;
  store.target = att || null;
  if (!same) store.emit("target", store.target);
}

// 생성 · 화질 개선 · 극점 보정 공통 제출 → 결과 칸이 진행을 확인
export async function submitJob(body, message) {
  const id = store.runId();
  if (!id) throw new Error("먼저 사진을 올리거나 작업을 여세요.");
  const result = await post("/api/generate", {run_id: id, ...body});
  if (store.isRun(id)) store.emit("job", {runId: id, n: result.attempt, message: message(result)});
  return result;
}

// 새 사진 올린 직후: 그 작업을 엶
export async function afterUpload(uploaded) {
  await openRun(uploaded.run_id);
  store.emit("runs:changed");
}
