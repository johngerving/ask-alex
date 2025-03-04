<script lang="ts" module>
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import * as Card from '$lib/components/ui/card/index.js';
	import { Messages } from '$lib/components/chat/Messages';
	import { MessageInput } from '$lib/components/chat/MessageInput';

	let messages = $state<Message[]>([]);

	const addMessage = (message: Message) => {
		messages.push(message);
	};
</script>

<Card.Root class="flex h-full w-full flex-col">
	<Card.Header>
		<Card.Title>Chat</Card.Title>
	</Card.Header>
	<Card.Content class="flex-grow overflow-auto">
		<Messages {messages} />
	</Card.Content>
	<Card.Footer class="w-full">
		<MessageInput {messages} {addMessage} />
	</Card.Footer>
</Card.Root>
