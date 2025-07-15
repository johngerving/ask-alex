<script lang="ts">
	import Messages from '$lib/components/chat/messages.svelte';
	import MessageInput from '$lib/components/chat/message-input.svelte';
	import type { Message } from '$lib/types/message';
	import type { Tool } from '$lib/types/toolCall';
	import { cn } from '$lib/utils';
	import { page } from '$app/state';
	import { fade } from 'svelte/transition';

	let { chatId, chatHistory = $bindable() }: { chatId: string; chatHistory: Message[] } = $props();

	let showHome = $derived(page.url.pathname === '/chat');
</script>

<div class="flex h-full w-full flex-col items-center">
	<div class="flex min-h-0 w-full max-w-4xl flex-grow flex-col p-4">
		<div class={cn('min-h-0 w-full p-0', !showHome ? 'item-grow' : '')}>
			<Messages {chatHistory} />
		</div>
		<div class="relative m-auto h-fit w-full p-0">
			{#if showHome}
				<h2 class="absolute -top-20 w-full text-center text-3xl">How can I help you?</h2>
			{/if}
			<MessageInput {chatId} bind:chatHistory />
		</div>
	</div>
</div>

<style>
	.item-grow {
		flex-grow: 20;
	}
</style>
