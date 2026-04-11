export declare abstract class EnvService {
    abstract getEnvValue(key: string): Promise<string | null>;
    abstract setEnvValue(key: string, value: string): Promise<void>;
}
//# sourceMappingURL=service.env.d.ts.map