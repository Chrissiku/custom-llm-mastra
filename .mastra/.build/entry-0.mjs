import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { MastraModelGateway } from '@mastra/core/llm';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

"use strict";
const weatherTool = createTool({
  id: "get-weather",
  description: "Get current weather for a location",
  inputSchema: z.object({
    location: z.string().describe("City name")
  }),
  outputSchema: z.object({
    output: z.string()
  }),
  execute: async () => {
    return {
      output: "The weather is sunny"
    };
  }
});

"use strict";
const PRIVATE_GATEWAY_ID = "private";
const PRIVATE_PROVIDER_ID = "my-provider";
const PRIVATE_DEFAULT_MODEL = "kimi-k2.5:cloud";
function privateChatModel(modelId = PRIVATE_DEFAULT_MODEL) {
  return `${PRIVATE_GATEWAY_ID}/${PRIVATE_PROVIDER_ID}/${modelId}`;
}
function normalizeOpenAiCompatibleBaseUrl(raw) {
  let url = raw.trim().replace(/\/$/, "");
  if (url.endsWith("/chat/completions")) {
    url = url.slice(0, -"/chat/completions".length).replace(/\/$/, "");
  }
  return url;
}
class MyPrivateGateway extends MastraModelGateway {
  id = PRIVATE_GATEWAY_ID;
  name = "My Private Gateway";
  async fetchProviders() {
    const raw = process.env.CUSTOM_URL;
    if (!raw) {
      throw new Error(
        "Missing CUSTOM_URL. Use the base URL before /chat/completions, e.g. http://localhost:3000/api"
      );
    }
    const baseUrl = normalizeOpenAiCompatibleBaseUrl(raw);
    return {
      [PRIVATE_PROVIDER_ID]: {
        name: "My Provider",
        models: [PRIVATE_DEFAULT_MODEL],
        apiKeyEnvVar: "CUSTOM_API_KEY",
        gateway: this.id,
        url: baseUrl
      }
    };
  }
  buildUrl(modelId, envVars) {
    const raw = envVars.CUSTOM_URL ?? process.env.CUSTOM_URL;
    if (!raw) {
      throw new Error(
        `No base URL configured for model: ${modelId}. Set CUSTOM_URL in your environment.`
      );
    }
    return normalizeOpenAiCompatibleBaseUrl(raw);
  }
  async getApiKey(modelId) {
    const apiKey = process.env.CUSTOM_API_KEY;
    if (!apiKey) {
      throw new Error(
        `Missing CUSTOM_API_KEY environment variable for model: ${modelId}. Set CUSTOM_API_KEY in your environment.`
      );
    }
    return apiKey;
  }
  async resolveLanguageModel({
    modelId,
    providerId,
    apiKey
  }) {
    const baseURL = this.buildUrl(`${providerId}/${modelId}`, {});
    const includeUsage = process.env.CUSTOM_OPENAI_INCLUDE_USAGE === "true" || process.env.CUSTOM_OPENAI_INCLUDE_USAGE === "1";
    return createOpenAICompatible({
      name: "private-llm",
      apiKey,
      baseURL,
      transformRequestBody: (body) => body.stream === true ? body : { ...body, stream: false },
      supportsStructuredOutputs: false,
      includeUsage
    }).chatModel(modelId);
  }
}

"use strict";
const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions: `
      You are a helpful weather assistant that provides accurate weather information.

      Your primary function is to help users get weather details for specific locations. When responding:
      - Always ask for a location if none is provided
      - If the location name isn't in English, please translate it
      - If giving a location with multiple parts (e.g. "New York, NY"), use the most relevant part (e.g. "New York")
      - Include relevant details like humidity, wind conditions, and precipitation
      - Keep responses concise but informative

      Use the weatherTool to fetch current weather data.
`,
  model: `${PRIVATE_GATEWAY_ID}/${PRIVATE_PROVIDER_ID}/${PRIVATE_DEFAULT_MODEL}`,
  tools: { weatherTool }
});

"use strict";
const mastra = new Mastra({
  gateways: {
    [`${PRIVATE_GATEWAY_ID}Gateway`]: new MyPrivateGateway()
  },
  agents: {
    weatherAgent
  }
});
mastra.addGateway(new MyPrivateGateway(), PRIVATE_GATEWAY_ID);

export { mastra };
