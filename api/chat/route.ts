import { streamText } from 'ai';
import { google } from '@ai-sdk/google';

export async function POST(req: Request) {
  const body = await req.json();

  const rawMessages = body?.messages ?? [];
  const vitResult = body?.data?.vitResult;

  // 🔥 Convert UIMessage → ModelMessage
  const messages = rawMessages.map((msg: any) => {
    if (Array.isArray(msg.parts)) {
      return {
        role: msg.role,
        content: msg.parts
          .filter((p: any) => p.type === 'text')
          .map((p: any) => p.text)
          .join(' ')
      };
    }

    return {
      role: msg.role,
      content: typeof msg.content === 'string' ? msg.content : ''
    };
  });

  const systemPrompt = `
You are an AI vs Real image detector and forensic expert.

If an internal Vision Transformer classification result is provided,
use it as ground truth.

Label: ${vitResult?.prediction ?? 'unknown'}
Confidence: ${vitResult?.confidence ?? 'unknown'}%

Explain naturally why the image may be AI-generated or real based on classification result.
Keep under 120 words.
`;

  const result = streamText({
    model: google('gemini-2.5-flash'),
    system: systemPrompt,
    messages
  });

  return result.toUIMessageStreamResponse();
}