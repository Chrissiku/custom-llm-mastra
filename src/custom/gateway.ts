import {
  MastraModelGateway,
  type GatewayLanguageModel,
  type ProviderConfig,
} from '@mastra/core/llm';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createOpenAiCompatibleStudioFetch } from './openai-compatible-fetch';

/**
 * AI SDK posts to `{baseURL}/chat/completions`.
 * Your server exposes `POST http://localhost:3000/api/chat/completions`,
 * so baseURL must be `http://localhost:3000/api` (not `.../api/v1`).
 */
function normalizeOpenAiCompatibleBaseUrl(raw: string): string {
  let url = raw.trim().replace(/\/$/, '');
  if (url.endsWith('/chat/completions')) {
    url = url.slice(0, -'/chat/completions'.length).replace(/\/$/, '');
  }
  return url;
}

/** OpenAI-compatible endpoint (e.g. local proxy, private deployment). */
export class MyPrivateGateway extends MastraModelGateway {
  /** Prefix for model IDs: `private/my-provider/<model>`. */
  readonly id = 'private';

  readonly name = 'My Private Gateway';

  async fetchProviders(): Promise<Record<string, ProviderConfig>> {
    const raw = process.env.CUSTOM_URL;
    if (!raw) {
      throw new Error(
        'Missing CUSTOM_URL. Use the base URL before /chat/completions, e.g. http://localhost:3000/api',
      );
    }
    const baseUrl = normalizeOpenAiCompatibleBaseUrl(raw);

    return {
      'my-provider': {
        name: 'My Provider',
        models: ['gemma3:1b'],
        apiKeyEnvVar: 'CUSTOM_API_KEY',
        gateway: this.id,
        url: baseUrl,
      },
    };
  }

  buildUrl(
    modelId: string,
    envVars: Record<string, string>,
  ): string | undefined {
    const raw = envVars.CUSTOM_URL ?? process.env.CUSTOM_URL;
    if (!raw) {
      throw new Error(
        `No base URL configured for model: ${modelId}. Set CUSTOM_URL in your environment.`,
      );
    }
    return normalizeOpenAiCompatibleBaseUrl(raw);
  }

  async getApiKey(modelId: string): Promise<string> {
    const apiKey = process.env.CUSTOM_API_KEY;
    if (!apiKey) {
      throw new Error(
        `Missing CUSTOM_API_KEY environment variable for model: ${modelId}. Set CUSTOM_API_KEY in your environment.`,
      );
    }
    return apiKey;
  }

  async resolveLanguageModel({
    modelId,
    providerId,
    apiKey,
  }: {
    modelId: string;
    providerId: string;
    apiKey: string;
  }): Promise<GatewayLanguageModel> {
    const baseURL = this.buildUrl(`${providerId}/${modelId}`, {})!;

    const includeUsage =
      process.env.CUSTOM_OPENAI_INCLUDE_USAGE === 'true' ||
      process.env.CUSTOM_OPENAI_INCLUDE_USAGE === '1';

    return createOpenAICompatible({
      // Shown in traces / Studio; distinct from gateway model id prefix.
      name: 'private-llm',
      apiKey,
      baseURL,
      fetch: createOpenAiCompatibleStudioFetch(),
      // doGenerate() omits `stream`; some proxies default to SSE, which breaks JSON parsing in the AI SDK.
      transformRequestBody: (body) =>
        body.stream === true ? body : { ...body, stream: false },
      // Local servers often reject json_schema; Studio still works with tools via normal tool_calls.
      supportsStructuredOutputs: false,
      // stream_options: { include_usage: true } breaks many OpenAI-compatible servers during Studio streaming.
      includeUsage,
    }).chatModel(modelId);
  }
}