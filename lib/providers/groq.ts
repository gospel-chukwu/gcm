import Groq from 'groq-sdk';
import { handleError } from '../../utils/errors.js';
import { getConfig, saveConfig } from '../config.js';

// Non-chat model patterns to skip (audio, safety, embedding)
const SKIP_PATTERNS = ['whisper', 'guard', 'audio', 'tts', 'embed', 'orpheus'];

// Preferred chat model families in priority order
const PREFERRED_PATTERNS = [
  'openai/gpt-oss-120b',
  'openai/gpt-oss',
  'qwen',
  'llama',
];

async function getBestAvailableModel(client: Groq): Promise<string> {
  const models = await client.models.list();

  const chatIds: string[] = (models.data ?? [])
    .map((m: any) => m.id as string)
    .filter(id => !SKIP_PATTERNS.some(p => id.toLowerCase().includes(p)));

  for (const pattern of PREFERRED_PATTERNS) {
    const match = chatIds.find(id => id.toLowerCase().includes(pattern));
    if (match) return match;
  }

  return chatIds[0] ?? 'openai/gpt-oss-20b';
}

export async function validateKey(apiKey: string): Promise<boolean> {
  try {
    const client = new Groq({ apiKey });
    await client.models.list();
    return true;
  } catch (err: any) {
    handleError(err);
  }
}

export async function generate(
  prompt: string,
  apiKey: string,
  model: string,
): Promise<string> {
  const client = new Groq({ apiKey });

  try {
    const response = await client.chat.completions.create({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.3,
    });
    return response.choices[0]?.message?.content?.trim() ?? '';
  } catch (err: any) {
    // Model deprecated or not found — auto-recover with the best live model
    if (err?.status === 404) {
      try {
        const fallbackModel = await getBestAvailableModel(client);

        const response = await client.chat.completions.create({
          model: fallbackModel,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.3,
        });

        // Silently update config so future runs use the working model
        try {
          const config = getConfig();
          saveConfig({ ...config, model: fallbackModel });
        } catch {
          // Non-fatal: generation succeeded even if config update fails
        }

        return response.choices[0]?.message?.content?.trim() ?? '';
      } catch (retryErr: any) {
        handleError(retryErr);
      }
    }

    handleError(err);
  }
}
