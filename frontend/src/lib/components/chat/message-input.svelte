<script lang="ts">
	import type { Message } from '$lib/types/message';
	import { MessageStatus, MessageUpdateType } from '$lib/types/message';

	import { Button } from '$lib/components/ui/button';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';
	import MaterialSymbolsArrowUpwardRounded from '~icons/material-symbols/arrow-upward-rounded';
	import MaterialSymbolsStop from '~icons/material-symbols/stop';
	import { sendMessages } from '$lib/utils/chat/sendMessages';

	import { v4 } from 'uuid';
	import { goto, invalidateAll, replaceState } from '$app/navigation';
	import type { Tool } from '$lib/types/toolCall';
	import { createChat } from '$lib/utils/chat/createChat';
	import { page } from '$app/state';

	let { chatId, chatHistory = $bindable() }: { chatId: string; chatHistory: Message[] } = $props();

	let text = $state('');

	// Disable the submit button if there is no message content
	let messageEmpty: boolean = $derived(text.length == 0);
	let messageInProgress: boolean = $derived(false);

	function handleOnKeyDown(e: KeyboardEvent) {
		// Send the message if the user presses enter but is not pressing shift.
		// If they are pressing shift, a newline is created.
		if (!e.shiftKey && e.code == 'Enter') {
			e.preventDefault();
			handleSend();
		}
	}

	async function handleSend() {
		let url = page.url;
		let currentChatId = chatId;
		if (url.pathname === '/chat') {
			const newChatId = await createChat();
			console.log('New chat created with ID:', newChatId);
			currentChatId = newChatId;
			goto(`/chat/${currentChatId}`);
		}

		let userMessageId = v4();
		let assistantMessageId = v4();

		chatHistory = [
			...chatHistory,
			{
				id: userMessageId,
				content: text,
				reasoning: '',
				toolCalls: [],
				role: 'user',
				status: MessageStatus.Finished
			}
		];

		sendMessages(chatHistory, currentChatId, {
			onStart: () => {
				messageInProgress = true;

				chatHistory = [
					...chatHistory,
					{
						id: assistantMessageId,
						content: '',
						reasoning: '',
						toolCalls: [],
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
			onToolCall: (toolCall: Tool) => {
				chatHistory = chatHistory.map((message) => {
					if (message.id === assistantMessageId) {
						return {
							...message,
							toolCalls: [...message.toolCalls, toolCall]
						};
					}
					return message;
				});
			},
			onFinish: (response: string) => {
				messageInProgress = false;

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
				messageInProgress = false;

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

<form
	onsubmit={handleSend}
	class="focus-within:outline-primary border-input w-full overflow-hidden rounded-md border shadow-sm outline-1 focus-within:outline hover:cursor-text"
>
	<label for="message-input" class="flex h-full w-full flex-col items-end hover:cursor-text">
		<ChatTextarea
			class="m-1 h-11 min-h-11 resize-none px-5 py-2 placeholder:text-gray-500"
			id="message-input"
			spellcheck="false"
			bind:value={text}
			placeholder="Ask Alex"
			onkeydown={handleOnKeyDown}
		/>
		<Button
			type="submit"
			disabled={messageEmpty || messageInProgress}
			class="relative mb-1 mr-1 h-11 w-11 overflow-hidden rounded-full transition-all hover:cursor-default"
		>
			<MaterialSymbolsArrowUpwardRounded class="absolute h-3/4 w-3/4 text-xl" />
		</Button>
	</label>
</form>
