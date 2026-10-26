// 한 작업의 조회가 끝난 뒤 다음 조회를 예약한다. 화면 전환 후 늦게 온 응답은 버린다.
// 조회가 실패하면 멈추지 않고 간격을 늘려(최대 15초) 다시 시도한다 — 버튼이 묶이지 않음.
export function createPoller(fetchStatus, onStatus, onError, delay = 1500, maxDelay = 15000) {
  let timer = null;
  let controller = null;
  let version = 0;

  function stop() {
    version++;
    clearTimeout(timer);
    controller?.abort();
    controller = null;
  }

  function start(runId, attempt) {
    stop();
    const current = version;
    controller = new AbortController();
    const signal = controller.signal;
    let wait = delay;
    async function tick() {
      if (current !== version) return;
      let terminal = false;
      try {
        const status = await fetchStatus(runId, attempt, signal);
        if (current !== version) return;
        wait = delay;
        terminal = status.state === "done" || status.state === "error";
        onStatus(status, attempt);
      } catch (error) {
        if (current !== version) return;
        wait = Math.min(maxDelay, wait * 2);
        onError(error, wait);
      }
      if (!terminal && current === version) timer = setTimeout(tick, wait);
    }
    timer = setTimeout(tick, delay);
  }

  return {start, stop};
}
