import { createWatcher } from '@mastra/deployer/build';
import { Bundler } from '@mastra/deployer/bundler';
export declare class DevBundler extends Bundler {
    private customEnvFile?;
    constructor(customEnvFile?: string);
    getEnvFiles(): Promise<string[]>;
    prepare(outputDirectory: string): Promise<void>;
    watch(entryFile: string, outputDirectory: string, toolsPaths: (string | string[])[]): ReturnType<typeof createWatcher>;
    bundle(): Promise<void>;
}
//# sourceMappingURL=DevBundler.d.ts.map