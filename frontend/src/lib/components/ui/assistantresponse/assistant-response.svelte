<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { cn } from '$lib/utils';
	import { marked } from 'marked';
	import SvgSpinners90RingWithBg from '~icons/svg-spinners/90-ring-with-bg';

	let { message }: { message: Message } = $props();
</script>

<div class={cn('flex w-max max-w-[75%] flex-col gap-2 px-3 py-3')}>
	{#if message.status === 'waiting'}
		<SvgSpinners90RingWithBg class="text-primary h-8 w-8" />
	{:else}
		{@html marked(message.content)}
	{/if}
</div>
