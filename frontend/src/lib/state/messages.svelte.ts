import {
	MessageStatus,
	MessageUpdateType,
	type Message,
	type MessageUpdate
} from '$lib/types/message';

let messages = $state<Message[]>([]);

function updateMessageWithId(id: string, updateFn: (message: Message) => void) {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].id === id) {
			updateFn(messages[i]);
			break;
		}
	}
}

export const messageStore = {
	get messages() {
		return messages;
	},
	set messages(newMessages: Message[]) {
		messages = newMessages;
	},
	createMessage(message: Message) {
		messages.push(message);
	},
	updateMessage(id: string, update: MessageUpdate) {
		switch (update.type) {
			case MessageUpdateType.Delta:
				updateMessageWithId(id, (message) => {
					message.content += update.delta || '';
					// message.content = message.content.replaceAll(' .', '.');
				});
				break;
			case MessageUpdateType.ReasoningDelta:
				updateMessageWithId(id, (message) => {
					message.reasoning += update.delta || '';
				});
				break;
			case MessageUpdateType.FinalAnswer:
				updateMessageWithId(id, (message) => {
					message.content = update.text;
					message.status = MessageStatus.Finished;
				});
				break;
			case MessageUpdateType.Error:
				updateMessageWithId(id, (message) => {
					message.status = MessageStatus.Error;
				});
				break;
		}
	}
};
