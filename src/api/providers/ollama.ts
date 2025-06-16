import { Anthropic } from "@anthropic-ai/sdk"
import { Message, Ollama } from "ollama"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOllamaMessages } from "../transform/ollama-format"
import { ApiStream } from "../transform/stream"
import { withRetry } from "../retry"

export class OllamaHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: Ollama

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new Ollama({ host: this.options.ollamaBaseUrl || "http://localhost:11434" })
	}

	@withRetry({ retryAllErrors: true })
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const ollamaMessages: Message[] = [{ role: "system", content: systemPrompt }, ...convertToOllamaMessages(messages)]

		try {
			// Create a promise that rejects after timeout
			const timeoutMs = this.options.requestTimeoutMs || 60000
			const timeoutPromise = new Promise<never>((_, reject) => {
				setTimeout(() => reject(new Error(`Ollama request timed out after ${timeoutMs / 1000} seconds`)), timeoutMs)
			})

			// Handle streaming vs non-streaming requests
			const useStreaming = this.options.ollamaUseStreaming ?? true

			if (useStreaming) {
				// Create streaming API request
				const apiPromise = this.client.chat({
					model: this.getModel().id,
					messages: ollamaMessages,
					stream: true,
					options: {
						num_ctx: Number(this.options.ollamaApiOptionsCtxNum) || 32768,
					},
				})

				// Race the API request against the timeout
				const stream = await Promise.race([apiPromise, timeoutPromise])
				console.log("Ollama connection established successfully")

				try {
					for await (const chunk of stream) {
						if (typeof chunk.message.content === "string") {
							yield {
								type: "text",
								text: chunk.message.content,
							}
						}

						// Handle token usage if available
						if (chunk.eval_count !== undefined || chunk.prompt_eval_count !== undefined) {
							yield {
								type: "usage",
								inputTokens: chunk.prompt_eval_count || 0,
								outputTokens: chunk.eval_count || 0,
							}
						}
					}
				} catch (streamError: any) {
					console.error("Error processing Ollama stream:", streamError)
					throw new Error(`Ollama stream processing error: ${streamError.message || "Unknown error"}`)
				}
			} else {
				// Create non-streaming API request
				const apiPromise = this.client.chat({
					model: this.getModel().id,
					messages: ollamaMessages,
					stream: false,
					options: {
						num_ctx: Number(this.options.ollamaApiOptionsCtxNum) || 32768,
					},
				})

				// Race the API request against the timeout
				const response = await Promise.race([apiPromise, timeoutPromise])
				console.log("Ollama connection established successfully")

				// Yield the complete response as a single chunk
				if (typeof response.message.content === "string") {
					yield {
						type: "text",
						text: response.message.content,
					}
				}

				// Handle token usage if available
				if (response.eval_count !== undefined || response.prompt_eval_count !== undefined) {
					yield {
						type: "usage",
						inputTokens: response.prompt_eval_count || 0,
						outputTokens: response.eval_count || 0,
					}
				}
			}
		} catch (error: any) {
			// Check if it's a timeout error
			if (error.message && error.message.includes("timed out")) {
				const timeoutMs = this.options.requestTimeoutMs || 30000
				throw new Error(`Ollama request timed out after ${timeoutMs / 1000} seconds`)
			}

			// Enhance error reporting
			const statusCode = error.status || error.statusCode
			const errorMessage = error.message || "Unknown error"

			console.error(`Ollama API error (${statusCode || "unknown"}): ${errorMessage}`)
			throw error
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.ollamaModelId || "",
			info: this.options.ollamaApiOptionsCtxNum
				? { ...openAiModelInfoSaneDefaults, contextWindow: Number(this.options.ollamaApiOptionsCtxNum) || 32768 }
				: openAiModelInfoSaneDefaults,
		}
	}
}
