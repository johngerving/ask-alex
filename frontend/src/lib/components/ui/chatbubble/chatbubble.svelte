<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';
	import { cn } from '$lib/utils';
	import { marked } from 'marked';
	import { fly, fade } from 'svelte/transition';

	let { message }: { message: Message } = $props();

	// Display message differently if message is a user message vs. assistant message
	let isUser: boolean = $derived(message.type === MessageType.User);
</script>

<div
	class={cn(
		'flex w-max max-w-[75%] flex-col gap-2 rounded-lg px-3 py-2 text-sm',
		isUser ? 'bg-muted ml-auto' : 'bg-muted'
	)}
>
	{message.content}
</div>

<!-- <div>
	<div class={`chat-bubble flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
		<div
			class={`leading-1.5 flex w-full max-w-xl flex-col rounded-b-xl border-gray-200 bg-gray-100 p-4 ${isUser ? 'rounded-l-xl' : 'rounded-r-xl'}`}
		>
			<div class="flex items-center space-x-2 rtl:space-x-reverse">
				<span class="text-sm font-semibold text-gray-900">
					{#if isUser}
						User
					{:else}
						ALEX
					{/if}
				</span>
			</div>
			<div class="py-2.5 text-sm font-normal text-gray-900">
				{@html marked(message.content)}
			</div>
		</div>
	</div>
</div> -->
