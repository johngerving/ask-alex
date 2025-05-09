<script lang="ts">
	import { AssistantResponse } from '$lib/components/ui/assistantresponse';
	import { ChatBubble } from '$lib/components/ui/chatbubble';
	import { scrollToBottom } from '$lib/utils/chat/scrollToBottom';
	import { messageStore } from '$lib/state/messages.svelte';

	$effect(() => {
		// Scroll to the bottom of the messages when a new one is added
		scrollToBottom(ref, messageStore.messages);
	});

	let ref: Element;
</script>

<div
	bind:this={ref}
	use:scrollToBottom={messageStore.messages}
	class="[&::-webkit-scrollbar-track]:bg-background
    flex
    h-full w-full flex-col
	gap-6 overflow-y-auto
    [&::-webkit-scrollbar-thumb]:rounded-full
    [&::-webkit-scrollbar-thumb]:bg-zinc-200
    dark:[&::-webkit-scrollbar-thumb]:bg-neutral-500
    [&::-webkit-scrollbar-track]:rounded-full
    dark:[&::-webkit-scrollbar-track]:bg-neutral-700
    [&::-webkit-scrollbar]:w-2
    "
	style="scrollbar-gutter: stable"
>
	{#each messageStore.messages as message}
		{#if message.role === 'assistant'}
			<AssistantResponse {message} />
		{:else}
			<ChatBubble {message} />
		{/if}
	{/each}
</div>
