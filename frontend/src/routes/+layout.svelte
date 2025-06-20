<script lang="ts">
	import '../app.css';
	import { PUBLIC_BACKEND_URL } from '$env/static/public';
	let { children } = $props();
</script>

{@render children()}
