<script lang="ts" module>
	import * as Card from '$lib/components/ui/card/index.js';
	import { Messages } from '$lib/components/chat/Messages';
	import { MessageInput } from '$lib/components/chat/MessageInput';
	import { messageStore } from '$lib/state/messages.svelte';
</script>

<div class="flex h-full w-full flex-col items-center">
	<div class="flex min-h-0 w-full max-w-4xl flex-grow flex-col p-4">
		<div class="min-h-0 w-full flex-grow p-0">
			<Messages />
		</div>
		<div class="h-fit w-full p-0">
			<MessageInput />
		</div>
	</div>
</div>
