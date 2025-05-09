<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { marked } from 'marked';

	let { message }: { message: Message } = $props();
</script>

<div class={'bg-muted ml-auto flex w-max max-w-[75%] flex-col gap-2 rounded-lg px-5 py-3'}>
	{@html marked(message.content)}
</div>
