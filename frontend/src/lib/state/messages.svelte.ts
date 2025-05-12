import {
	MessageStatus,
	MessageUpdateType,
	type Message,
	type MessageUpdate
} from '$lib/types/message';

let messages = $state<Message[]>([]);

export const messageStore = {
	get messages() {
		return messages;
	},
	createMessage(message: Message) {
		messages.push(message);
	},
	updateMessage(id: string, update: MessageUpdate) {
		if (update.type === MessageUpdateType.Delta) {
			for (let i = messages.length - 1; i >= 0; i--) {
				if (messages[i].id === id) {
					messages[i].content += update.delta || '';
					// messages[i].content = messages[i].content.replaceAll(' .', '.');
					break;
				}
			}
		} else if (update.type === MessageUpdateType.FinalAnswer) {
			for (let i = messages.length - 1; i >= 0; i--) {
				if (messages[i].id === id) {
					messages[i].content = update.text;
					messages[i].status = MessageStatus.Finished;
					break;
				}
			}
		} else if (update.type === MessageUpdateType.Error) {
			for (let i = messages.length - 1; i >= 0; i--) {
				if (messages[i].id === id) {
					messages[i].status = MessageStatus.Error;
					break;
				}
			}
		}
	}
};
