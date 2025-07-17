<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import { PUBLIC_BACKEND_URL } from '$env/static/public';
</script>

<main class="flex h-full w-full items-center justify-center">
	<Card.Root class="m-auto w-96 py-4 text-center">
		<Card.Header>
			<Card.Title class="text-3xl">Welcome to Ask Alex Humboldt</Card.Title>
		</Card.Header>
		<Card.Content class="">
			<p class="mb-12 text-lg">Please sign in to continue</p>
			<div class="flex items-center justify-center dark:bg-gray-800">
				<form action={`${PUBLIC_BACKEND_URL}/auth/login`} method="post">
					<button
						class="dark:hover:border-primary hover:border-primary border-muted dark:border-muted flex gap-2 rounded-lg border px-6 py-3 text-slate-700 transition duration-300 hover:text-slate-900 hover:shadow dark:text-slate-200 dark:hover:text-slate-300"
					>
						<img
							class="h-6 w-6"
							src="https://www.svgrepo.com/show/475656/google-color.svg"
							loading="lazy"
							alt="google logo"
						/>
						<span>Login with Google</span>
					</button>
				</form>
			</div>
		</Card.Content>
	</Card.Root>
</main>
