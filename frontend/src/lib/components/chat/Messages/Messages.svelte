<script lang="ts">
	import { AssistantResponse } from '$lib/components/ui/assistantresponse';
	import { ChatBubble } from '$lib/components/ui/chatbubble';
	import { MessageType, type Message } from '$lib/types/message';
	import { scrollToBottom } from '$lib/utils/chat/scrollToBottom';

	let { messages }: { messages: Message[] } = $props();

	$effect(() => {
		// Scroll to the bottom of the messages when a new one is added
		scrollToBottom(ref, messages);
	});

	let ref: Element;
</script>

<div
	bind:this={ref}
	use:scrollToBottom={messages}
	class="flex
    h-full
    w-full flex-col gap-2
    overflow-y-auto rounded-lg
    border
    px-2
    py-4
    [&::-webkit-scrollbar-thumb]:rounded-full
    [&::-webkit-scrollbar-thumb]:bg-gray-300
    dark:[&::-webkit-scrollbar-thumb]:bg-neutral-500
    [&::-webkit-scrollbar-track]:rounded-full
    [&::-webkit-scrollbar-track]:bg-gray-100
    dark:[&::-webkit-scrollbar-track]:bg-neutral-700
    [&::-webkit-scrollbar]:w-2
    "
	style="scrollbar-gutter: stable"
>
	{#each messages as message}
		{#if message.type === MessageType.Assistant}
			<AssistantResponse {message} />
		{:else}
			<ChatBubble {message} />
		{/if}
	{/each}
</div>
