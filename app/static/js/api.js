// Thin client for the Unfacet API.
export class ApiError extends Error {}

async function handle(res) {
  if (res.ok) return res.json();
  let msg = `${res.status} ${res.statusText}`;
  try { const j = await res.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch {}
  throw new ApiError(msg);
}

export const api = {
  async get(path) { return handle(await fetch(`/api${path}`)); },
  async post(path, body = {}) {
    return handle(await fetch(`/api${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));
  },
  async upload(file) {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return handle(await fetch('/api/sessions', { method: 'POST', body: fd }));
  },
  async job(jid, onProgress) {
    // poll a background job until it finishes
    let delay = 150;
    for (;;) {
      const j = await api.get(`/jobs/${jid}`);
      onProgress?.(j.progress);
      if (j.status === 'done') return j.result;
      if (j.status === 'error') throw new ApiError(j.error || 'The job failed.');
      await new Promise(r => setTimeout(r, delay));
      delay = Math.min(delay * 1.4, 700);
    }
  },
};

export function b64ToArray(s, Type) {
  const bin = atob(s);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}
