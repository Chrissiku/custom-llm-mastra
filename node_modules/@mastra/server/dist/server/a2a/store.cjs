'use strict';

// src/server/a2a/store.ts
var InMemoryTaskStore = class {
  store = /* @__PURE__ */ new Map();
  activeCancellations = /* @__PURE__ */ new Set();
  async load({ agentId, taskId }) {
    const entry = this.store.get(`${agentId}-${taskId}`);
    if (!entry) {
      return null;
    }
    return { ...entry };
  }
  async save({ agentId, data }) {
    const key = `${agentId}-${data.id}`;
    if (!data.id) {
      throw new Error("Task ID is required");
    }
    this.store.set(key, { ...data });
  }
};

exports.InMemoryTaskStore = InMemoryTaskStore;
//# sourceMappingURL=store.cjs.map
//# sourceMappingURL=store.cjs.map