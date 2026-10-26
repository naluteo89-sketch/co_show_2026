// 한 작업의 조회가 끝난 뒤 다음 조회를 예약한다. 화면 전환 후 늦게 온 응답은 버린다.
export function createPoller(fetchStatus, onStatus, onError, delay = 1500) {
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
    async function tick() {
      if (current !== version) return;
      let terminal = false;
      try {
        const status = await fetchStatus(runId, attempt, signal);
        if (current !== version) return;
        terminal = status.state === "done" || status.state === "error";
        onStatus(status, attempt);
      } catch (error) {
        if (current !== version) return;
        onError(error);
      }
      if (!terminal && current === version) timer = setTimeout(tick, delay);
    }
    timer = setTimeout(tick, delay);
  }

  return {start, stop};
}
