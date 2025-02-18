<script lang="ts">
	import * as Card from '$lib/components/ui/card/index.js';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';

	let height = $state();
</script>

<main class="h-full w-full p-3">
	<Card.Root class="h-full w-full">
		<Card.Header>
			<Card.Title>Card Title</Card.Title>
			<Card.Description>Card Description</Card.Description>
		</Card.Header>
		<Card.Content>
			<p>Card Content</p>
		</Card.Content>
		<Card.Footer>
			<ChatTextarea class="h-11 min-h-11 resize-none" />
		</Card.Footer>
	</Card.Root>
</main>
