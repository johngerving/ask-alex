import type { v4 } from 'uuid';
import type { Tool } from '$lib/types/toolCall';

export type Message = {
	role: 'user' | 'assistant';
	reasoning: string;
	toolCalls: Tool[];
	content: string;
	id: string;
	status: MessageStatus;
};

export type MessageUpdate =
	| MessageDelta
	| MessageReasoningDelta
	| MessageError
	| MessageFinalAnswer;

export enum MessageUpdateType {
	Error = 'status',
	Delta = 'delta',
	ReasoningDelta = 'reasoningDelta',
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

export interface MessageReasoningDelta {
	type: MessageUpdateType.ReasoningDelta;
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
