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
		'flex w-max max-w-[75%] flex-col gap-2 rounded-lg px-3 py-3',
		isUser ? 'bg-muted ml-auto' : 'bg-muted'
	)}
>
	{@html marked(message.content)}
</div>