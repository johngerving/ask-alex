<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';
	import MaterialSymbolsArrowUpwardRounded from '~icons/material-symbols/arrow-upward-rounded';
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

		sendMessages(messages, {
			onStart: () => {
				messages.push({
					content: '',
					type: MessageType.Assistant,
					id: assistantMessageId,
					status: 'waiting'
				});
			},
			onUpdate: (response: string) => {
				console.log(response);
				for (let i = messages.length - 1; i >= 0; i--) {
					if (messages[i].id === assistantMessageId) {
						messages[i].content = response;
						messages[i].status = 'done';
						break;
					}
				}
			},
			onFinish: () => {},
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
	<Button type="submit" disabled={sendDisabled} class="h-11 w-11 rounded-full p-0 transition-all">
		<MaterialSymbolsArrowUpwardRounded class="m-2 h-full w-full text-xl" />
	</Button>
</form>
