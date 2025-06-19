import { Anthropic } from "@anthropic-ai/sdk"
import { Message, Ollama } from "ollama"
import { Agent, setGlobalDispatcher } from "undici"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOllamaMessages } from "../transform/ollama-format"
import { ApiStream, ApiStreamChunk } from "../transform/stream"

/**
 * OllamaHandler – minimal, type‑safe implementation that avoids undici’s
 * default 5‑minute body‑timeout without custom fetch wrappers.
 *
 * Strategy:
 *  1. Set a *global* undici Agent with `headersTimeout` and `bodyTimeout`
 *     disabled – this affects every fetch in the process.
 *  2. Add a manual Promise‑race timeout so callers can still limit request
 *     duration (we simply stop awaiting and throw; we don't abort the fetch).
 *  3. Expose `client` so unit‑tests can stub `chat()` easily.
 *
 * Retry Strategy Considerations:
 *  - Automatic retries are deliberately disabled for streaming chat interactions
 *  - Reasons:
 *    1. Non-idempotent streams: Retries could restart chat, duplicate tokens
 *    2. Ambiguous failure modes: Retries might mask underlying connection issues
 *    3. Prefer explicit back-pressure handling by caller
 *  - Potential future retry strategy could use selective retry for specific network errors
 */

// Disable undici’s built‑in timeouts globally
setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 }))

export class OllamaHandler implements ApiHandler {
	/** Exposed for test stubbing – e.g. sinon.stub(handler.client, "chat") */
	public readonly client: Ollama

	constructor(private readonly options: ApiHandlerOptions) {
		this.client = new Ollama({
			host: this.options.ollamaBaseUrl ?? "http://localhost:11434",
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.options.ollamaModelId
		if (!modelId) throw new Error("Ollama model id is required")

		const ollamaMessages: Message[] = [{ role: "system", content: systemPrompt }, ...convertToOllamaMessages(messages)]

		// Manual per‑request timeout (default 30 s)
		const timeoutMs = this.options.requestTimeoutMs ?? 30_000
		const timeoutPromise = new Promise<never>((_, reject) =>
			setTimeout(() => reject(new Error(`Ollama request timed out after ${timeoutMs / 1000} seconds`)), timeoutMs),
		)

		// Request streaming
		const apiPromise = this.client.chat({
			model: modelId,
			messages: ollamaMessages,
			stream: true,
		})

		let stream
		try {
			stream = await Promise.race([apiPromise, timeoutPromise])
		} catch (err: any) {
			// “fetch failed” and similar become clearer diagnostics
			const cause: any = err?.cause ?? err
			if (cause?.code) {
				throw new Error(
					`Could not reach Ollama at ${
						this.options.ollamaBaseUrl ?? "http://localhost:11434"
					} — ${cause.code}: ${cause.message ?? cause}`,
				)
			}

			const statusCode = err?.status ?? err?.statusCode ?? "unknown"
			throw new Error(`Ollama API error (${statusCode}): ${err?.message ?? err}`)
		}

		// Process streaming chunks
		for await (const chunk of stream) {
			if (typeof chunk.message.content === "string") {
				yield { type: "text", text: chunk.message.content }
			}

			if (chunk.eval_count !== undefined || chunk.prompt_eval_count !== undefined) {
				const usageData: ApiStreamChunk = {
					type: "usage",
					inputTokens: chunk.prompt_eval_count ?? 0,
					outputTokens: chunk.eval_count ?? 0,
				}
				yield usageData
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		const id = this.options.ollamaModelId! // validated above
		return {
			id,
			info: this.options.ollamaApiOptionsCtxNum
				? {
						...openAiModelInfoSaneDefaults,
						contextWindow: Number(this.options.ollamaApiOptionsCtxNum) || 32_768,
					}
				: openAiModelInfoSaneDefaults,
		}
	}
}
