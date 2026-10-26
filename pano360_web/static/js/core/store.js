// 화면 상태 한 곳 + 알림. 컴포넌트끼리는 서로를 직접 부르지 않고 이걸로만 주고받는다.
//
//   cfg       서버 설정 (/api/config)
//   run       지금 연 작업 (meta.json) · 없으면 null
//   attempts  그 작업의 시도 목록
//   target    5·6단계가 쓸 결과 (완료된 시도 하나) — 사용자가 고르고, 새 시도가 실패해도 그대로
//
// 알림: "cfg" "run" "attempts" "target" "analysis" "reuse" "job" "event" "runs:changed"

const listeners = {};

export const store = {
  cfg: null,
  run: null,
  attempts: [],
  target: null,
  targetPinned: false,     // 사용자가 직접 고른 결과면 새 결과가 나와도 바꾸지 않음
  services: {},            // 컴포넌트가 내놓는 작은 기능 (예: 마스크 저장)

  on(name, fn) { (listeners[name] = listeners[name] || []).push(fn); },
  emit(name, data) { (listeners[name] || []).forEach(fn => { try { fn(data); } catch (e) { console.error(e); } }); },

  runId() { return this.run ? this.run.run_id : null; },
  isRun(id) { return !!this.run && this.run.run_id === id; },
};

export const doneAttempts = attempts => (attempts || []).filter(a => a.state === "done" && (a.files || []).length);
