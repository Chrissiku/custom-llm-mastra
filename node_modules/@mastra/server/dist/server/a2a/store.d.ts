import type { Task } from '@mastra/core/a2a';
export declare class InMemoryTaskStore {
    private store;
    activeCancellations: Set<string>;
    load({ agentId, taskId }: {
        agentId: string;
        taskId: string;
    }): Promise<Task | null>;
    save({ agentId, data }: {
        agentId: string;
        data: Task;
    }): Promise<void>;
}
//# sourceMappingURL=store.d.ts.map