// 实验 worker 只允许连接本次回环代理；真实凭据仅由 Python 父进程持有。
// 不改产品消息、提示词或返回；意外联网直接失败并留下日志。
const allowed = value => {
  const url = new URL(typeof value === 'string' || value instanceof URL ? value : value.url);
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) {
    console.error('EVAL_NETWORK_DENIED', url.origin);
    throw new Error('实验 worker 禁止非回环网络');
  }
};
const originalFetch = globalThis.fetch;
globalThis.fetch = function(input, ...args) {
  allowed(input);
  return originalFetch.call(this, input, ...args);
};
for (const protocol of ['http', 'https']) {
  const module = require('node:' + protocol);
  for (const name of ['request', 'get']) {
    const original = module[name];
    module[name] = function(input, ...args) {
      if (typeof input === 'string' || input instanceof URL) allowed(input);
      else allowed(`${protocol}://${input.hostname || input.host || 'localhost'}`);
      return original.call(this, input, ...args);
    };
  }
}
