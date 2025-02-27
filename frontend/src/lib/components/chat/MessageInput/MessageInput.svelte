<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';
	import { sendMessages } from '$lib/utils/chat/sendMessages';

	let { messages, addMessage }: { messages: Message[]; addMessage: (m: Message) => void } =
		$props();

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

		const responsePromise = sendMessages(messages);

		// Reset the input content
		text = '';

		// Wait for assistant to respond
		const assistantResponse = await responsePromise;

		// Add message with assistant response
		addMessage(assistantResponse);
	}
</script>

<form onsubmit={handleSend} class="flex w-full items-end gap-2">
	<ChatTextarea class="h-11 min-h-11 resize-none" bind:value={text} onkeydown={handleOnKeyDown} />
	<Button type="submit" disabled={sendDisabled} class="h-11 w-11 p-0">
		<MaterialSymbolsSendOutlineRounded class="text-xl" />
	</Button>
</form>
