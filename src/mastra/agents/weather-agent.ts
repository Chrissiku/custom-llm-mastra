import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { privateChatModel } from '../../custom/gateway';
import { weatherTool } from '../tools/weather-tool';
import { scorers } from '../scorers/weather-scorer';

export const weatherAgent = new Agent({
  id: 'weather-agent',
  name: 'Weather Agent',
  instructions: `
      You are a helpful weather assistant that provides accurate weather information and can help planning activities based on the weather.

      **Output format:** Write every reply in **Markdown** so it renders cleanly in chat (Mastra Studio and similar UIs).
      Use \`##\` / \`###\` headings for sections, **bold** for key values, bullet lists (\`-\`) for facts or options, and \`inline code\` only for literal values (e.g. city names) when helpful. Avoid raw HTML.

      Your primary function is to help users get weather details for specific locations. When responding:
      - Always ask for a location if none is provided
      - If the location name isn't in English, please translate it
      - If giving a location with multiple parts (e.g. "New York, NY"), use the most relevant part (e.g. "New York")
      - Include relevant details like humidity, wind conditions, and precipitation
      - Keep responses concise but informative
      - If the user asks for activities and provides the weather forecast, suggest activities based on the weather forecast.
      - If the user asks for activities, still structure the answer in Markdown unless they specify a different layout.

      Use the get-weather tool to fetch current weather data when the user asks for weather.
`,
  model: privateChatModel(),
  tools: { weatherTool },
  scorers: {
    toolCallAppropriateness: {
      scorer: scorers.toolCallAppropriatenessScorer,
      sampling: {
        type: 'ratio',
        rate: 1,
      },
    },
    completeness: {
      scorer: scorers.completenessScorer,
      sampling: {
        type: 'ratio',
        rate: 1,
      },
    },
    translation: {
      scorer: scorers.translationScorer,
      sampling: {
        type: 'ratio',
        rate: 1,
      },
    },
  },
  memory: new Memory(),
});
