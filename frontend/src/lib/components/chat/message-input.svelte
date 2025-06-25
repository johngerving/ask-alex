<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageStatus, MessageUpdateType } from '$lib/types/message';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsSendOutlineRounded from '~icons/material-symbols/send-outline-rounded';
	import MaterialSymbolsArrowUpwardRounded from '~icons/material-symbols/arrow-upward-rounded';
	import { sendMessages } from '$lib/utils/chat/sendMessages';

	import { v4 } from 'uuid';
	import { invalidateAll } from '$app/navigation';

	let { chatId, chatHistory = $bindable() }: { chatId: number; chatHistory: Message[] } = $props();

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
		// messageStore.createMessage({
		// 	content: text,
		// 	reasoning: '',
		// 	role: 'user',
		// 	id: userMessageId,
		// 	status: MessageStatus.Finished
		// });

		console.log('Sending message:', text);
		// chatHistory.push({
		// 	id: userMessageId,
		// 	content: text,
		// 	reasoning: '',
		// 	role: 'user',
		// 	status: MessageStatus.Finished
		// });
		chatHistory = [
			...chatHistory,
			{
				id: userMessageId,
				content: text,
				reasoning: '',
				role: 'user',
				status: MessageStatus.Finished
			}
		];

		sendMessages(chatHistory, chatId, {
			onStart: () => {
				chatHistory = [
					...chatHistory,
					{
						id: assistantMessageId,
						content: '',
						reasoning: '',
						role: 'assistant',
						status: MessageStatus.Started
					}
				];
			},
			onUpdateContent: (delta: string) => {
				chatHistory = chatHistory.map((message) => {
					if (message.id === assistantMessageId) {
						return {
							...message,
							content: message.content + delta
						};
					}
					return message;
				});
			},
			onUpdateReasoning: (delta: string) => {
				chatHistory = chatHistory.map((message) => {
					if (message.id === assistantMessageId) {
						return {
							...message,
							reasoning: message.reasoning + delta
						};
					}
					return message;
				});
			},
			onFinish: (response: string) => {
				chatHistory = chatHistory.map((message) => {
					if (message.id === assistantMessageId) {
						return {
							...message,
							content: response,
							status: MessageStatus.Finished
						};
					}
					return message;
				});

				invalidateAll(); // Invalidate all data to refresh the chat history
			},
			onError: (error: string) => {
				console.error(error);
				chatHistory = chatHistory.map((message) => {
					if (message.id === assistantMessageId) {
						return {
							...message,
							content: error,
							status: MessageStatus.Error
						};
					}
					return message;
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
	<Button
		type="submit"
		disabled={sendDisabled}
		class="relative h-11 w-11 overflow-hidden rounded-full transition-all"
	>
		<MaterialSymbolsArrowUpwardRounded class="absolute h-3/4 w-3/4 text-xl" />
	</Button>
</form>
