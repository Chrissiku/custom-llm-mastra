import type { Message, Task, TaskStatus, TaskContext, TaskArtifactUpdateEvent } from '@mastra/core/a2a';
import type { IMastraLogger } from '@mastra/core/logger';
import type { InMemoryTaskStore } from './store.js';
export declare function applyUpdateToTask(current: Task, update: Omit<TaskStatus, 'timestamp'> | TaskArtifactUpdateEvent): Task;
export declare function loadOrCreateTask({ agentId, taskId, taskStore, message, contextId, metadata, logger, }: {
    agentId: string;
    taskId: string;
    taskStore: InMemoryTaskStore;
    message: Message;
    contextId?: string;
    metadata?: Record<string, unknown>;
    logger?: IMastraLogger;
}): Promise<Task>;
export declare function createTaskContext({ task, userMessage, history, activeCancellations, }: {
    task: Task;
    userMessage: Message;
    history: Message[];
    activeCancellations: Set<string>;
}): TaskContext;
//# sourceMappingURL=tasks.d.ts.map