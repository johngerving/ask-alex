<script lang="ts">
	import '../app.css';
	import { PUBLIC_BACKEND_URL } from '$env/static/public';
	let { children } = $props();
</script>

<div class="flex h-full w-full flex-col">
	<header class="flex justify-between px-6 py-4">
		<a href="/" class="text-2xl font-medium">
			<h1 class="text-2xl">Ask Alex Humboldt</h1>
		</a>
	</header>
	{@render children()}
</div>
