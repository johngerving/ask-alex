<script lang="ts" module>
	import type { Message } from '$lib/types/message';
	import { MessageType } from '$lib/types/message';

	import * as Card from '$lib/components/ui/card/index.js';
	import { Messages } from '$lib/components/chat/Messages';
	import { MessageInput } from '$lib/components/chat/MessageInput';

	let messages = $state<Message[]>([]);
</script>

<Card.Root class="flex h-full w-full flex-col items-center">
	<Card.Header class="h-11 w-full">
		<Card.Title>Ask Alex</Card.Title>
	</Card.Header>
	<div class="flex min-h-0 w-full max-w-4xl flex-grow flex-col gap-2 p-4">
		<Card.Content class="min-h-0 w-full flex-grow p-0">
			<Messages {messages} />
		</Card.Content>
		<Card.Footer class="h-fit w-full p-0">
			<MessageInput {messages} />
		</Card.Footer>
	</div>
</Card.Root>
