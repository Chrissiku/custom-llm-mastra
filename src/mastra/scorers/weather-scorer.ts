import { z } from 'zod';
import { createCompletenessScorer } from '@mastra/evals/scorers/prebuilt';
import {
  extractToolCalls,
  getAssistantMessageFromRunOutput,
  getUserMessageFromRunInput,
} from '@mastra/evals/scorers/utils';
import { createScorer } from '@mastra/core/evals';

/** Must match `createTool({ id })` in `weather-tool.ts` (model-facing tool name). */
const WEATHER_TOOL = 'get-weather';

function checkToolOrder(
  actualTools: string[],
  expectedOrder: string[],
  strictMode: boolean,
): boolean {
  if (strictMode) {
    return JSON.stringify(actualTools) === JSON.stringify(expectedOrder);
  }
  const expectedIndices: number[] = [];
  for (const name of expectedOrder) {
    const index = actualTools.indexOf(name);
    if (index === -1) return false;
    expectedIndices.push(index);
  }
  for (let i = 1; i < expectedIndices.length; i++) {
    const current = expectedIndices[i];
    const prev = expectedIndices[i - 1];
    if (current !== undefined && prev !== undefined && current <= prev) {
      return false;
    }
  }
  return true;
}

function scoreToolCallAccuracy({
  expectedTool,
  actualTools,
  strictMode,
  expectedToolOrder,
}: {
  expectedTool: string | undefined;
  actualTools: string[];
  strictMode: boolean;
  expectedToolOrder: string[] | undefined;
}): number {
  if (actualTools.length === 0) return 0;
  if (expectedToolOrder?.length) {
    return checkToolOrder(actualTools, expectedToolOrder, strictMode) ? 1 : 0;
  }
  if (!expectedTool) return 0;
  if (strictMode) {
    return actualTools.length === 1 && actualTools[0] === expectedTool ? 1 : 0;
  }
  return actualTools.includes(expectedTool) ? 1 : 0;
}

/**
 * Same behavior as `createToolCallAccuracyScorerCode`, but does not throw when
 * input/output messages are missing. Mastra Studio often runs scorers while
 * warming the viewer (`No toolset ... waiting`) with an empty payload.
 */
export const toolCallAppropriatenessScorer = createScorer({
  id: 'code-tool-call-accuracy-scorer',
  name: 'Tool Call Accuracy Scorer',
  description: `Evaluates whether the LLM selected the correct tool (${WEATHER_TOOL}) from the available tools`,
  type: 'agent',
})
  .preprocess(async ({ run }) => {
    const missingMessages =
      !run.input?.inputMessages?.length || !run.output?.length;
    if (missingMessages) {
      return {
        skipped: true as const,
        skipReason:
          'Skipped: no input or output messages on this run (e.g. Studio viewer before a completed agent turn).',
      };
    }
    const { tools: actualTools } = extractToolCalls(run.output);
    return {
      skipped: false as const,
      expectedTool: WEATHER_TOOL,
      actualTools,
      strictMode: false,
      expectedToolOrder: undefined as string[] | undefined,
    };
  })
  .generateScore(({ results }) => {
    const p = results.preprocessStepResult;
    if (!p || p.skipped) return 1;
    return scoreToolCallAccuracy({
      expectedTool: p.expectedTool,
      actualTools: p.actualTools,
      strictMode: p.strictMode,
      expectedToolOrder: p.expectedToolOrder,
    });
  })
  .generateReason(({ results, score }) => {
    const p = results.preprocessStepResult;
    if (p?.skipped) return p.skipReason;
    return `Expected tool "${WEATHER_TOOL}"; called ${JSON.stringify(p?.actualTools ?? [])}. Score=${score}.`;
  });

export const completenessScorer = createCompletenessScorer();

// Custom LLM-judged scorer: evaluates if non-English locations are translated appropriately
export const translationScorer = createScorer({
  id: 'translation-quality-scorer',
  name: 'Translation Quality',
  description:
    'Checks that non-English location names are translated and used correctly',
  type: 'agent',
  judge: {
    model: 'private/my-provider/gemma3:1b',
    instructions:
      'You are an expert evaluator of translation quality for geographic locations. ' +
      'Determine whether the user text mentions a non-English location and whether the assistant correctly uses an English translation of that location. ' +
      'Be lenient with transliteration differences and diacritics. ' +
      'Return only the structured JSON matching the provided schema.',
  },
})
  .preprocess(({ run }) => {
    const userText = getUserMessageFromRunInput(run.input) || '';
    const assistantText = getAssistantMessageFromRunOutput(run.output) || '';
    return { userText, assistantText };
  })
  .analyze({
    description:
      'Extract location names and detect language/translation adequacy',
    outputSchema: z.object({
      nonEnglish: z.boolean(),
      translated: z.boolean(),
      confidence: z.number().min(0).max(1).default(1),
      explanation: z.string().default(''),
    }),
    createPrompt: ({ results }) => `
            You are evaluating if a weather assistant correctly handled translation of a non-English location.
            User text:
            """
            ${results.preprocessStepResult.userText}
            """
            Assistant response:
            """
            ${results.preprocessStepResult.assistantText}
            """
            Tasks:
            1) Identify if the user mentioned a location that appears non-English.
            2) If non-English, check whether the assistant used a correct English translation of that location in its response.
            3) Be lenient with transliteration differences (e.g., accents/diacritics).
            Return JSON with fields:
            {
            "nonEnglish": boolean,
            "translated": boolean,
            "confidence": number, // 0-1
            "explanation": string
            }
        `,
  })
  .generateScore(({ results }) => {
    const r = (results as any)?.analyzeStepResult || {};
    if (!r.nonEnglish) return 1; // If not applicable, full credit
    if (r.translated)
      return Math.max(0, Math.min(1, 0.7 + 0.3 * (r.confidence ?? 1)));
    return 0; // Non-English but not translated
  })
  .generateReason(({ results, score }) => {
    const r = (results as any)?.analyzeStepResult || {};
    return `Translation scoring: nonEnglish=${r.nonEnglish ?? false}, translated=${r.translated ?? false}, confidence=${r.confidence ?? 0}. Score=${score}. ${r.explanation ?? ''}`;
  });

export const scorers = {
  toolCallAppropriatenessScorer,
  completenessScorer,
  translationScorer,
};
