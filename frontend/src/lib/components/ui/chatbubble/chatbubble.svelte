<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';
	import { marked } from 'marked';
	import { fly, fade } from 'svelte/transition';

	let { message }: { message: Message } = $props();

	let content =
		'`asdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdfasdfjaskldf;asjdkfl;asdf`';

	// Display message differently if message is a user message vs. assistant message
	let isUser: boolean = $derived(message.type === MessageType.User);
</script>

<div>
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
			<div class=".chat-content whitespace-break-spaces py-2.5 text-sm font-normal text-gray-900">
				{@html marked(content)}
			</div>
		</div>
	</div>
</div>

<style>
	.chat-content {
		white-space: normal;
		overflow-x: scroll;
	}
	.chat-content :global(pre) {
		white-space: normal;
		background-color: blue;
		overflow-x: scroll;
	}

	.chat-content :global(code) {
		white-space: normal;
		overflow-x: scroll;
	}
</style>
