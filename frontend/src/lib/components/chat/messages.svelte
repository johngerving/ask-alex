<script lang="ts">
	import { AssistantResponse } from '$lib/components/ui/assistantresponse';
	import { ChatBubble } from '$lib/components/ui/chatbubble';
	import type { Message } from '$lib/types/message';
	import { scrollToBottom } from '$lib/utils/chat/scrollToBottom';

	let { chatHistory }: { chatHistory: Message[] } = $props();

	$effect(() => {
		// Scroll to the bottom of the messages when a new one is added
		scrollToBottom(ref, chatHistory);
	});

	let ref: Element;
</script>

<div
	bind:this={ref}
	use:scrollToBottom={chatHistory}
	class="[&::-webkit-scrollbar-track]:bg-background
    gradientback
    flex h-full w-full
	flex-col gap-6
    overflow-y-auto pb-4
    pr-2
    [&::-webkit-scrollbar-thumb]:rounded-full
    [&::-webkit-scrollbar-thumb]:bg-zinc-200
    dark:[&::-webkit-scrollbar-thumb]:bg-neutral-500
    [&::-webkit-scrollbar-track]:rounded-full
	dark:[&::-webkit-scrollbar-track]:bg-neutral-700
	[&::-webkit-scrollbar]:w-2
    "
	style="scrollbar-gutter: stable"
>
	{#each chatHistory as message}
		{#if message.role === 'assistant'}
			<AssistantResponse {message} />
		{:else}
			<ChatBubble {message} />
		{/if}
	{/each}
</div>

<style>
	.gradientback {
		-webkit-mask: linear-gradient(#000 0%, #000 calc(100% - 40px), #0000 100%);
		mask: linear-gradient(#000 0%, #000 calc(100% - 40px), #0000 100%);
	}
</style>
