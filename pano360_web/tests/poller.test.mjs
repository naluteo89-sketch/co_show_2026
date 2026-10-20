import assert from "node:assert/strict";
import {test} from "node:test";
import {createPoller} from "../static/poller.js";

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return {promise, resolve};
};
async function until(check) {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await sleep(5);
  }
  assert.fail("condition timed out");
}

test("switching runs discards an old response and aborts its request", async () => {
  const old = deferred(), seen = [], signals = [];
  const poller = createPoller((id, n, signal) => {
    signals.push(signal);
    return id === "old" ? old.promise : Promise.resolve({state: "done", id});
  }, result => seen.push(result.id), assert.fail, 1);
  try {
    poller.start("old", 1);
    await until(() => signals.length === 1);
    poller.start("new", 2);
    await until(() => seen.length === 1);
    old.resolve({state: "done", id: "old"});
    await sleep(10);
    assert.deepEqual(seen, ["new"]);
    assert.equal(signals[0].aborted, true);
  } finally { poller.stop(); }
});

test("slow requests never overlap; terminal status stops polling", async () => {
  const first = deferred();
  let calls = 0;
  const seen = [];
  const poller = createPoller(() => ++calls === 1 ? first.promise : Promise.resolve({state: "done"}),
    result => seen.push(result.state), assert.fail, 1);
  try {
    poller.start("run", 1);
    await until(() => calls === 1);
    await sleep(15);
    assert.equal(calls, 1);
    first.resolve({state: "running"});
    await until(() => seen.includes("done"));
    await sleep(10);
    assert.equal(calls, 2);
    assert.deepEqual(seen, ["running", "done"]);
  } finally { poller.stop(); }
});

test("transient errors retry; stop also suppresses pending response handlers", async () => {
  let calls = 0, errors = 0, updates = 0;
  const pending = deferred();
  const poller = createPoller(async () => {
    if (++calls === 1) throw new Error("offline");
    return pending.promise;
  }, () => updates++, () => errors++, 1);
  try {
    poller.start("run", 1);
    await until(() => calls === 2);
    poller.stop();
    pending.resolve({state: "done"});
    await sleep(10);
    assert.equal(errors, 1);
    assert.equal(updates, 0);
    assert.equal(calls, 2);
  } finally { poller.stop(); }
});
