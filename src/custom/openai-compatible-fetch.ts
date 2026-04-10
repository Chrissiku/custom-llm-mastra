/**
 * Wraps fetch so OpenAI-compatible responses work with Mastra Studio / AI SDK:
 * - Non-stream (`doGenerate`): normalizes completion JSON; if the server still returns SSE, aggregates
 *   chunks into one `chat.completion` JSON (avoids "Invalid JSON response").
 * - Stream: some gateways accept `stream: true` but still return `application/json` with a full
 *   completion object. The SDK expects `text/event-stream` (SSE). We synthesize SSE from that JSON.
 */

function flattenMessageContent(content: unknown): string | null {
  if (content == null) return null;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part == null) return '';
        if (typeof part === 'string') return part;
        if (typeof part === 'object' && part !== null && 'text' in part) {
          const t = (part as { text?: unknown }).text;
          return typeof t === 'string' ? t : '';
        }
        return '';
      })
      .join('');
  }
  if (typeof content === 'object' && content !== null && 'text' in content) {
    const t = (content as { text?: unknown }).text;
    return typeof t === 'string' ? t : null;
  }
  return null;
}

function normalizeToolCallArguments(message: Record<string, unknown>): void {
  const toolCalls = message.tool_calls;
  if (!Array.isArray(toolCalls)) return;
  for (const tc of toolCalls) {
    if (!tc || typeof tc !== 'object') continue;
    const fn = (tc as { function?: { arguments?: unknown } }).function;
    if (!fn || typeof fn !== 'object') continue;
    const args = fn.arguments;
    if (args != null && typeof args !== 'string') {
      fn.arguments = JSON.stringify(args);
    }
  }
}

function requestBodyToString(body: BodyInit | null | undefined): string | null {
  if (body == null) return null;
  if (typeof body === 'string') return body;
  if (body instanceof Uint8Array) {
    return new TextDecoder('utf-8').decode(body);
  }
  if (typeof ArrayBuffer !== 'undefined' && body instanceof ArrayBuffer) {
    return new TextDecoder('utf-8').decode(body);
  }
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(body)) {
    return body.toString('utf8');
  }
  return null;
}

function parseRequestStreamFlag(init?: RequestInit): boolean {
  const raw = requestBodyToString(init?.body ?? null);
  if (raw == null) return false;
  try {
    const parsed = JSON.parse(raw) as { stream?: boolean };
    return parsed.stream === true;
  } catch {
    return false;
  }
}

function isNonStreamCompletionShape(data: unknown): data is Record<string, unknown> {
  if (!data || typeof data !== 'object') return false;
  const choices = (data as { choices?: unknown }).choices;
  if (!Array.isArray(choices) || choices.length === 0) return false;
  const first = choices[0] as { message?: unknown; delta?: unknown };
  return (
    first != null &&
    typeof first === 'object' &&
    'message' in first &&
    first.message != null &&
    typeof first.message === 'object'
  );
}

/** Mutates completion JSON in place for the non-streaming SDK parser. */
function normalizeCompletionMessage(data: Record<string, unknown>): void {
  const choices = data.choices as Array<{ message?: Record<string, unknown> }>;
  const first = choices[0];
  if (!first?.message || typeof first.message !== 'object') return;
  const msg = first.message;
  normalizeToolCallArguments(msg);
  const reasoningRaw = msg.reasoning_content ?? msg.reasoning;
  const reasoningStr = typeof reasoningRaw === 'string' ? reasoningRaw : '';
  const flat = flattenMessageContent(msg.content);
  if ((flat == null || flat === '') && reasoningStr.length > 0) {
    msg.content = reasoningStr;
    return;
  }
  if (flat !== null && typeof msg.content !== 'string') {
    msg.content = flat;
  }
}

/**
 * Turn a full chat.completion JSON into an OpenAI-style SSE body (what Studio's stream client expects).
 */
function chatCompletionToSse(data: Record<string, unknown>): string {
  normalizeCompletionMessage(data);

  const id = String(data.id ?? 'chatcmpl-synthetic');
  const created =
    typeof data.created === 'number' ? data.created : Math.floor(Date.now() / 1000);
  const model = String(data.model ?? '');
  const choices = data.choices as Array<{
    message?: Record<string, unknown>;
    finish_reason?: string | null;
  }>;
  const first = choices[0];
  const msg = first?.message ?? {};
  const finishReason = first?.finish_reason ?? 'stop';
  const usage = data.usage;

  let text = flattenMessageContent(msg.content) ?? '';
  const reasoningRaw = msg.reasoning_content ?? msg.reasoning;
  const reasoningStr =
    typeof reasoningRaw === 'string' ? reasoningRaw : '';
  // Many UIs (incl. Studio) render the main assistant bubble from text deltas only.
  // If the gateway puts the answer in reasoning_* and leaves content empty, mirror it into text.
  if (text.length === 0 && reasoningStr.length > 0) {
    text = reasoningStr;
  }
  const toolCallsRaw = msg.tool_calls;
  const toolCalls = Array.isArray(toolCallsRaw) ? toolCallsRaw : [];
  const lines: string[] = [];

  const base = { id, object: 'chat.completion.chunk' as const, created, model };

  lines.push(
    `data: ${JSON.stringify({
      ...base,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant' as const },
          finish_reason: null,
        },
      ],
    })}\n\n`,
  );

  if (toolCalls.length > 0) {
    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i] as {
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      };
      const tcId = tc.id ?? `call_${i}`;
      const name = tc.function?.name ?? 'tool';
      const args = tc.function?.arguments ?? '{}';

      lines.push(
        `data: ${JSON.stringify({
          ...base,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: i,
                    id: tcId,
                    type: 'function',
                    function: { name, arguments: '' },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        })}\n\n`,
      );

      if (args.length > 0) {
        const chunkSize = 256;
        for (let o = 0; o < args.length; o += chunkSize) {
          const piece = args.slice(o, o + chunkSize);
          lines.push(
            `data: ${JSON.stringify({
              ...base,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: i,
                        function: { arguments: piece },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            })}\n\n`,
          );
        }
      }
    }
  }

  if (text.length > 0) {
    const chunkSize = 160;
    for (let i = 0; i < text.length; i += chunkSize) {
      const piece = text.slice(i, i + chunkSize);
      lines.push(
        `data: ${JSON.stringify({
          ...base,
          choices: [
            {
              index: 0,
              delta: { content: piece },
              finish_reason: null,
            },
          ],
        })}\n\n`,
      );
    }
  }

  const finalChunk: Record<string, unknown> = {
    ...base,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: toolCalls.length > 0 ? 'tool_calls' : finishReason,
      },
    ],
  };
  if (usage && typeof usage === 'object') {
    finalChunk.usage = usage;
  }
  lines.push(`data: ${JSON.stringify(finalChunk)}\n\n`);
  lines.push('data: [DONE]\n\n');

  return lines.join('');
}

function looksLikeJsonChatCompletion(body: string): boolean {
  const t = body.trimStart();
  if (t.startsWith('data:')) return false;
  return t.startsWith('{') && t.includes('"choices"');
}

/** True when the body looks like OpenAI-style SSE (doGenerate sometimes gets this if the server ignores stream:false). */
function looksLikeOpenAiSseBody(body: string): boolean {
  const t = body.trimStart();
  if (t.startsWith('data:')) return true;
  return /(^|\n)\r?\n?\s*data:\s*/m.test(body);
}

/**
 * Collapse streamed chat.completion.chunk SSE into one chat.completion object for AI SDK doGenerate().
 */
function aggregateOpenAiSseChunksToCompletion(
  sseText: string,
): Record<string, unknown> | null {
  type ToolAcc = { id?: string; name?: string; args: string };
  const toolByIndex = new Map<number, ToolAcc>();

  let id = 'chatcmpl-aggregated';
  let created = Math.floor(Date.now() / 1000);
  let model = '';
  let content = '';
  let reasoning = '';
  let finishReason: string | null = 'stop';
  let usage: Record<string, unknown> | undefined;
  let parsedEvents = 0;

  const lines = sseText.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;

    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      continue;
    }
    parsedEvents++;

    if (typeof chunk.id === 'string' && chunk.id.length > 0) id = chunk.id;
    if (typeof chunk.created === 'number') created = chunk.created;
    if (typeof chunk.model === 'string' && chunk.model.length > 0) {
      model = chunk.model;
    }
    if (chunk.usage != null && typeof chunk.usage === 'object') {
      usage = chunk.usage as Record<string, unknown>;
    }

    const choices = chunk.choices as Array<{
      delta?: Record<string, unknown>;
      finish_reason?: string | null;
    }>;
    const choice = Array.isArray(choices) ? choices[0] : undefined;
    if (!choice || typeof choice !== 'object') continue;

    if (choice.finish_reason != null && choice.finish_reason !== '') {
      finishReason = String(choice.finish_reason);
    }

    const delta = choice.delta;
    if (!delta || typeof delta !== 'object') continue;

    const dc = delta.content;
    if (typeof dc === 'string' && dc.length > 0) content += dc;

    const rc = delta.reasoning_content ?? delta.reasoning;
    if (typeof rc === 'string' && rc.length > 0) reasoning += rc;

    const dtc = delta.tool_calls;
    if (!Array.isArray(dtc)) continue;
    for (const tc of dtc) {
      if (!tc || typeof tc !== 'object') continue;
      const rawIdx = (tc as { index?: number }).index;
      const idx =
        typeof rawIdx === 'number' && !Number.isNaN(rawIdx)
          ? rawIdx
          : toolByIndex.size;
      let acc = toolByIndex.get(idx);
      if (!acc) {
        acc = { args: '' };
        toolByIndex.set(idx, acc);
      }
      const tcId = (tc as { id?: string }).id;
      if (typeof tcId === 'string' && tcId.length > 0) acc.id = tcId;
      const fn = (tc as { function?: { name?: string; arguments?: string } })
        .function;
      if (fn && typeof fn === 'object') {
        if (typeof fn.name === 'string' && fn.name.length > 0) {
          acc.name = fn.name;
        }
        if (typeof fn.arguments === 'string' && fn.arguments.length > 0) {
          acc.args += fn.arguments;
        }
      }
    }
  }

  if (parsedEvents === 0) return null;

  const sortedIndices = [...toolByIndex.keys()].sort((a, b) => a - b);
  const tool_calls =
    sortedIndices.length > 0
      ? sortedIndices.map((idx) => {
          const acc = toolByIndex.get(idx)!;
          return {
            id: acc.id ?? `call_${idx}`,
            type: 'function',
            function: {
              name: acc.name ?? 'tool',
              arguments: acc.args.length > 0 ? acc.args : '{}',
            },
          };
        })
      : undefined;

  const message: Record<string, unknown> = {
    role: 'assistant',
    content: content.length > 0 ? content : null,
  };
  if (reasoning.length > 0) {
    message.reasoning_content = reasoning;
  }
  if (tool_calls != null) {
    message.tool_calls = tool_calls;
  }

  const out: Record<string, unknown> = {
    id,
    object: 'chat.completion',
    created,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason:
          tool_calls != null && tool_calls.length > 0 ? 'tool_calls' : finishReason,
      },
    ],
  };
  if (usage != null) {
    out.usage = usage;
  }
  return out;
}

function sseHeadersFrom(response: Response): Headers {
  const h = new Headers();
  h.set('content-type', 'text/event-stream; charset=utf-8');
  h.set('cache-control', 'no-cache');
  const rid = response.headers.get('x-request-id');
  if (rid) h.set('x-request-id', rid);
  return h;
}

export function createOpenAiCompatibleStudioFetch(
  innerFetch: typeof globalThis.fetch = globalThis.fetch.bind(globalThis),
): typeof globalThis.fetch {
  return async (input, init) => {
    const response = await innerFetch(input, init);
    if (!response.ok) return response;

    const streamRequested = parseRequestStreamFlag(init);
    const ct = response.headers.get('content-type') ?? '';

    // --- Stream request + JSON body: gateway ignored streaming; synthesize SSE for the AI SDK ---
    if (streamRequested) {
      if (ct.includes('text/event-stream')) {
        return response;
      }
      const jsonLikely =
        ct.includes('application/json') || ct.includes('json');
      let bodyText: string;
      try {
        bodyText = await response.clone().text();
      } catch {
        return response;
      }
      if (jsonLikely || looksLikeJsonChatCompletion(bodyText)) {
        let data: unknown;
        try {
          data = JSON.parse(bodyText);
        } catch {
          return response;
        }
        if (isNonStreamCompletionShape(data)) {
          const sseBody = chatCompletionToSse(data);
          return new Response(sseBody, {
            status: response.status,
            statusText: response.statusText,
            headers: sseHeadersFrom(response),
          });
        }
      }
      return response;
    }

    // --- Non-stream (AI SDK doGenerate): response must be JSON matching OpenAI chat.completion ---
    let bodyText: string;
    try {
      bodyText = await response.clone().text();
    } catch {
      return response;
    }

    try {
      const data = JSON.parse(bodyText) as unknown;
      if (
        data &&
        typeof data === 'object' &&
        isNonStreamCompletionShape(data)
      ) {
        const obj = data as Record<string, unknown>;
        normalizeCompletionMessage(obj);
        return new Response(JSON.stringify(obj), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
    } catch {
      // Body is not JSON; try SSE aggregation below.
    }

    if (ct.includes('text/event-stream') || looksLikeOpenAiSseBody(bodyText)) {
      const completion = aggregateOpenAiSseChunksToCompletion(bodyText);
      if (completion != null) {
        normalizeCompletionMessage(completion);
        const h = new Headers(response.headers);
        h.set('content-type', 'application/json; charset=utf-8');
        return new Response(JSON.stringify(completion), {
          status: response.status,
          statusText: response.statusText,
          headers: h,
        });
      }
    }

    return response;
  };
}
