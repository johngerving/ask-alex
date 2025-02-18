<script lang="ts">
    let { message }: { message: Message } = $props();
</script>

<div>
    <div class={"chat-bubble w-full flex justify-end"}>
        <div class={"flex flex-col w-full max-w-xl leading-1.5 p-4 border-gray-200 bg-gray-100 rounded-l-xl rounded-b-xl"}>
            <div class="flex items-center space-x-2 rtl:space-x-reverse">
                <span class="text-sm font-semibold text-gray-900">User</span>
            </div>
            <div class="text-sm font-normal py-2.5 text-gray-900">
                {message.content} 
            </div>
        </div>
    </div>
</div>