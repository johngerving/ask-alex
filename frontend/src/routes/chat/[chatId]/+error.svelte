<script lang="ts">
	import { page } from '$app/state';
</script>

<main class="flex h-full w-full flex-col items-center justify-center gap-3">
	<h1 class="text-6xl">{page.status}</h1>
	<h2 class="text-2xl">{page?.error?.message}</h2>
</main>
