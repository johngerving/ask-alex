<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageStatus, MessageUpdateType } from '$lib/types/message';
	import { messageStore } from '$lib/state/messages.svelte';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';
	import MaterialSymbolsArrowUpwardRounded from '~icons/material-symbols/arrow-upward-rounded';
	import { sendMessages } from '$lib/utils/chat/sendMessages';

	import { v4 } from 'uuid';

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
		let userMessageId = v4();
		let assistantMessageId = v4();

		// Add a message
		messageStore.createMessage({
			content: text,
			reasoning: '',
			role: 'user',
			id: userMessageId,
			status: MessageStatus.Finished
		});

		sendMessages(messageStore.messages, {
			onStart: () => {
				messageStore.createMessage({
					content: '',
					reasoning: '',
					role: 'assistant',
					id: assistantMessageId,
					status: MessageStatus.Started
				});
			},
			onUpdateContent: (delta: string) => {
				messageStore.updateMessage(assistantMessageId, {
					type: MessageUpdateType.Delta,
					delta: delta
				});
			},
			onUpdateReasoning: (delta: string) => {
				messageStore.updateMessage(assistantMessageId, {
					type: MessageUpdateType.ReasoningDelta,
					delta: delta
				});
			},
			onFinish: (response: string) => {
				messageStore.updateMessage(assistantMessageId, {
					type: MessageUpdateType.FinalAnswer,
					text: response
				});
			},
			onError: (error: any) => {
				console.error(error);
				messageStore.updateMessage(assistantMessageId, {
					type: MessageUpdateType.Error,
					error: error.message
				});
			}
		});

		// Reset the input content
		text = '';
	}
</script>

<form onsubmit={handleSend} class="flex w-full items-end gap-2">
	<ChatTextarea
		class="h-11 min-h-11 resize-none focus-visible:ring-offset-0"
		spellcheck="false"
		bind:value={text}
		onkeydown={handleOnKeyDown}
	/>
	<Button type="submit" disabled={sendDisabled} class="h-11 w-11 rounded-full p-0 transition-all">
		<MaterialSymbolsArrowUpwardRounded class="m-2 h-full w-full text-xl" />
	</Button>
</form>
