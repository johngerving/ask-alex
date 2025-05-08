<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { Spinner } from '$lib/components/ui/spinner';
	import * as smd from 'streaming-markdown';

	let { message }: { message: Message } = $props();
	let messageContent = $derived(message.content);

	function markdown(node: HTMLElement, textContent: string) {
		const renderer = smd.default_renderer(node);
		let parser = smd.parser(renderer);

		let characterQueue: string[] = [];
		let isTyping = false;
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

		return {
			update(newTextContent: string) {
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

<div class="flex w-max max-w-[75%] flex-col gap-2 px-3 py-3">
	{#if message.status === 'waiting'}
		<Spinner />
	{/if}
	<div
		use:markdown={messageContent}
		class="markdown [&_a:hover]:underline [&_a]:text-blue-500"
	></div>
</div>
