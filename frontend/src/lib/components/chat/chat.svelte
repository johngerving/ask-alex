<script lang="ts">
	import Messages from '$lib/components/chat/messages.svelte';
	import MessageInput from '$lib/components/chat/message-input.svelte';
	import type { Message } from '$lib/types/message';
	import type { Tool } from '$lib/types/toolCall';

	let { chatId, chatHistory = $bindable() }: { chatId: number; chatHistory: Message[] } = $props();
</script>

<div class="flex h-full w-full flex-col items-center">
	<div class="flex min-h-0 w-full max-w-4xl flex-grow flex-col p-4">
		<div class="min-h-0 w-full flex-grow p-0">
			<Messages {chatHistory} />
		</div>
		<div class="h-fit w-full p-0">
			<MessageInput {chatId} bind:chatHistory />
		</div>
	</div>
</div>
