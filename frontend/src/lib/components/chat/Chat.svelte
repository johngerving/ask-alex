<script lang="ts" module>
	import * as Card from '$lib/components/ui/card/index.js';
	import { Messages } from '$lib/components/chat/Messages';
	import { MessageInput } from '$lib/components/chat/MessageInput';
	import { messageStore } from '$lib/state/messages.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import AppSidebar from '$lib/components/chat/app-sidebar.svelte';
</script>

<Sidebar.Provider style="">
	<AppSidebar />
	<Sidebar.Trigger class="m-2 h-10 w-10" />
	<div class="flex h-svh w-full flex-col items-center">
		<div class="flex min-h-0 w-full max-w-4xl flex-grow flex-col p-4">
			<div class="min-h-0 w-full flex-grow p-0">
				<Messages />
			</div>
			<div class="h-fit w-full p-0">
				<MessageInput />
			</div>
		</div>
	</div>
</Sidebar.Provider>
