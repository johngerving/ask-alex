<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';
	import { sendMessages } from '$lib/utils/chat/sendMessages';

	import { v4 as uuidv4 } from 'uuid';

	let { messages = $bindable<Message[]>() }: { messages: Message[] } = $props();

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
		let userMessageId = uuidv4();
		let assistantMessageId = uuidv4();

		// Add a message
		messages.push({
			content: text,
			type: MessageType.User,
			id: userMessageId
		});

		let responseMessageSet = false;

		sendMessages(messages, {
			onStart: () => {
				messages.push({
					content: '',
					type: MessageType.Assistant,
					id: assistantMessageId
				});
			},
			onUpdate: (response: string) => {
				for (let i = messages.length - 1; i >= 0; i--) {
					if (messages[i].id === assistantMessageId) {
						messages[i].content = response;
						break;
					}
				}
			},
			onFinish: (response: string) => {
				for (let i = messages.length - 1; i >= 0; i--) {
					if (messages[i].id === assistantMessageId) {
						messages[i].content = response;
						break;
					}
				}
				console.log('finish', response);
			},
			onError: (error: any) => {
				console.log('error', error);
			}
		});

		// Reset the input content
		text = '';
	}
</script>

<form onsubmit={handleSend} class="flex w-full items-end gap-2">
	<ChatTextarea
		class="h-11 min-h-11 resize-none focus-visible:ring-offset-0"
		bind:value={text}
		onkeydown={handleOnKeyDown}
	/>
	<Button type="submit" disabled={sendDisabled} class="h-11 w-11 p-0 transition-all">
		<MaterialSymbolsSendOutlineRounded class="text-xl" />
	</Button>
</form>
