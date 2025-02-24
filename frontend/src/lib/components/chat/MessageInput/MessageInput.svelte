<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import { PUBLIC_RAG_ENDPOINT } from '$env/static/public';
	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';

	let { addMessage }: { addMessage: (m: Message) => void } = $props();

	let text = $state('');

	// Disable the submit button if there is no message content
	let sendDisabled: boolean = $derived(text.length == 0);

	function handleOnKeyDown(e: KeyboardEvent) {
		// Send the message if the user presses enter but is not pressing shift.
		// If they are pressing shift, a newline is created.
		if (!e.shiftKey && e.code == 'Enter') {
			e.preventDefault();
			handleSend();
		}
	}

	async function handleSend() {
		// Add a message
		addMessage({
			content: text,
			type: MessageType.User
		});

		const res = fetch(PUBLIC_RAG_ENDPOINT, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ query: text })
		});

		// Reset the input content
		text = '';

		const json = await (await res).json();
		const response = json.response;

		addMessage({
			content: response,
			type: MessageType.Assistant
		});
	}
</script>

<form onsubmit={handleSend} class="flex w-full items-end gap-2">
	<ChatTextarea class="h-11 min-h-11 resize-none" bind:value={text} onkeydown={handleOnKeyDown} />
	<Button type="submit" disabled={sendDisabled} class="h-11 w-11 p-0">
		<MaterialSymbolsSendOutlineRounded style="scale: 2" />
	</Button>
</form>
