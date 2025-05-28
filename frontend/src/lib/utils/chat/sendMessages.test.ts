import { describe, it, expect } from 'vitest';
import { parseData } from './sendMessages';

describe('parseData', () => {
	it('should parse a regular JSON string with "v" property', () => {
		const jsonData = '{"v": "hello world"}';
		expect(parseData(jsonData)).toBe('hello world');
	});

	it('should parse JSON string where "v" property value contains quotes', () => {
		const jsonData = '{"v": "hello \\"quoted\\" world"}';
		expect(parseData(jsonData)).toBe('hello "quoted" world');
	});

	it('should parse JSON string where "v" property value contains newline characters', () => {
		const jsonData = '{"v": "hello\\nworld"}';
		expect(parseData(jsonData)).toBe('hello\nworld');
	});

	it('should parse JSON string where "v" property value contains both quotes and newline characters', () => {
		const jsonData = '{"v": "hello \\"quoted\\"\\nand newlined world"}';
		expect(parseData(jsonData)).toBe('hello "quoted"\nand newlined world');
	});

	it('should parse JSON string with various special characters in "v" property', () => {
		const jsonData = '{"v": "value with backslash \\\\, tab \\t, and unicode \\u0041"}';
		expect(parseData(jsonData)).toBe('value with backslash \\, tab \t, and unicode A');
	});

	it('should throw an error if "v" property is missing', () => {
		const jsonData = '{"otherProperty": "some value"}';
		expect(() => parseData(jsonData)).toThrowError(
			'Data missing "v" property or is not a valid object'
		);
	});

	it('should throw an error for malformed JSON string', () => {
		const jsonData = '{"v": "unterminated string';
		expect(() => parseData(jsonData)).toThrowError(/^Invalid JSON string:.*/);
	});

	it('should throw an error if the input is not a JSON object string', () => {
		const jsonData = '"just a string"'; // Valid JSON, but not an object
		expect(() => parseData(jsonData)).toThrowError(
			'Data missing "v" property or is not a valid object'
		);
	});

	it('should throw an error if "v" property is present but not a string', () => {
		const jsonData = '{"v": 123}'; // 'v' is a number
		expect(() => parseData(jsonData)).toThrowError('"v" property is not a string');
	});

	it('should handle empty string value for "v"', () => {
		const jsonData = '{"v": ""}';
		expect(parseData(jsonData)).toBe('');
	});

	it('should handle JSON string with leading/trailing whitespace around the structure', () => {
		const jsonData = '  {"v": "trimmed"}  ';
		expect(parseData(jsonData)).toBe('trimmed');
	});
});
