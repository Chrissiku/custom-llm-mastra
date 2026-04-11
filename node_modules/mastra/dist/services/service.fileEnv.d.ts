import { EnvService } from './service.env.js';
export declare class FileEnvService extends EnvService {
    private filePath;
    constructor(filePath: string);
    private readFile;
    private writeFile;
    private updateEnvData;
    getEnvValue(key: string): Promise<string | null>;
    setEnvValue(key: string, value: string): Promise<void>;
}
//# sourceMappingURL=service.fileEnv.d.ts.map