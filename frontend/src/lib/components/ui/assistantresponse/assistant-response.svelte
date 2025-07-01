<script lang="ts">
	import { MessageStatus, type Message } from '$lib/types/message';
	import { Spinner } from '$lib/components/ui/spinner';
	import { ToolCall } from '$lib/components/ui/toolcall';
	import * as smd from 'streaming-markdown';
	import { marked } from 'marked';
	import { fade } from 'svelte/transition';

	let { message }: { message: Message } = $props();
	let messageContent = $derived(message.content);

	let isTyping = $state(false);

	function markdown(node: HTMLElement, textContent: string) {
		const renderer = smd.default_renderer(node);
		let parser = smd.parser(renderer);

		let characterQueue: string[] = [];
		let currentTimeoutId: ReturnType<typeof setTimeout> | undefined;
		let previousText = ''; // Track fully typed text

		// Process characters in queue and write them to the markdown parser
		function processQueue() {
			if (characterQueue.length === 0) {
				isTyping = false;
				return;
			}

			isTyping = true;

			// Add some randomness to the typing effect to make it look more natural
			const rand = Math.random();
			const chunkLength = Math.floor(rand * 20) + 1;
			const delay = Math.floor(rand * 80) + 10;

			// Remove a chunk of characters from the queue and write them to the parser
			const chars = characterQueue.splice(0, chunkLength).join('');
			smd.parser_write(parser, chars);

			currentTimeoutId = setTimeout(processQueue, delay); // Wait to write the next chunk
		}

		// Initial setup when the action is first applied or textContent is initially set
		if (textContent) {
			for (const char of textContent) {
				characterQueue.push(char);
			}
			previousText = textContent;
		}

		processQueue();

		return {
			update(newTextContent: string) {
				if (message.status === MessageStatus.Finished) {
					return;
				}
				if (currentTimeoutId) {
					clearTimeout(currentTimeoutId); // Clear pending timeout before processing new update
					isTyping = false; // Reset typing flag
				}

				let textToAdd = '';

				if (newTextContent.startsWith(previousText)) {
					// It's an addition to the existing text
					textToAdd = newTextContent.substring(previousText.length);
				} else {
					// The text has changed completely, not just an append.
					// Reset the display and queue.
					node.innerHTML = '';
					smd.parser_end(parser); // Finalize the old parser instance
					parser = smd.parser(smd.default_renderer(node)); // Re-initialize parser for the new content
					characterQueue = []; // Clear any pending characters from the old text
					textToAdd = newTextContent; // The entire new text needs to be typed
				}

				for (const char of textToAdd) {
					characterQueue.push(char);
				}
				previousText = newTextContent;

				if (!isTyping && characterQueue.length > 0) {
					processQueue(); // Start processing the queue if not already typing
				}
			},
			destroy() {
				if (currentTimeoutId) {
					clearTimeout(currentTimeoutId);
				}
				characterQueue = [];
				smd.parser_end(parser);
			}
		};
	}
</script>

<div class="w-[75%] px-3">
	<div class="">
		<div class="mb-3 flex flex-col gap-1">
			{#each message.toolCalls as toolCall (toolCall.id)}
				<ToolCall {toolCall} />
			{/each}
		</div>
		{#if message.status === MessageStatus.Started || isTyping}
			<div
				use:markdown={messageContent}
				class="markdown fade-in-chunks [&_a:hover]:underline [&_a]:text-blue-500"
			></div>
		{:else if message.status === MessageStatus.Finished}
			<div class="markdown [&_a:hover]:underline [&_a]:text-blue-500">
				{@html marked(message.content)}
			</div>
		{:else if message.status === MessageStatus.Error}
			<div class="text-destructive">{message.content}</div>
		{/if}
	</div>

	{#if message.status === MessageStatus.Started || isTyping}
		<div
			transition:fade={{ duration: 200 }}
			class="pulsing to-primary h-5 w-5 rounded-full bg-gradient-to-br from-green-400 via-emerald-500 shadow-md"
		></div>
	{/if}
</div>

<style>
	@keyframes pulse-scale {
		0% {
			transform: scale(1);
		}
		50% {
			transform: scale(1.1);
		}
		100% {
			transform: scale(1);
		}
	}

	.pulsing {
		display: block;
		animation: pulse-scale 1.2s ease-in-out infinite;
		transform-origin: center center;
	}
</style>
