import type { v4 } from 'uuid';

export type Message = {
	role: 'user' | 'assistant';
	reasoningContent: string;
	content: string;
	id: string;
	status: MessageStatus;
};

export type MessageUpdate = MessageDelta | MessageError | MessageFinalAnswer;

export enum MessageUpdateType {
	Error = 'status',
	Delta = 'delta',
	FinalAnswer = 'finalAnswer'
}

export enum MessageStatus {
	Started = 'started',
	Error = 'error',
	Finished = 'finished'
}

export interface MessageDelta {
	type: MessageUpdateType.Delta;
	delta: string;
}

export interface MessageError {
	type: MessageUpdateType.Error;
	error: string;
}

export interface MessageFinalAnswer {
	type: MessageUpdateType.FinalAnswer;
	text: string;
}
