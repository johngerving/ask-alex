<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { cn } from '$lib/utils';
	import { marked } from 'marked';
	import { Spinner } from '$lib/components/ui/spinner';

	let { message }: { message: Message } = $props();
	let messageContent = $derived(message.content);
</script>

<div
	class={cn(
		'flex w-max max-w-[75%] flex-col gap-2 px-3 py-3 [&_a:hover]:underline [&_a]:text-blue-500'
	)}
>
	{#if message.status === 'waiting'}
		<Spinner />
	{:else}
		{@html marked(messageContent)}
	{/if}
</div>
