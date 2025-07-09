ALTER TABLE documents ADD COLUMN text TEXT;
CREATE INDEX documents_text_idx ON documents USING gin(to_tsvector('english', text));

---- create above / drop below ----

DROP INDEX documents_text_idx;
ALTER TABLE documents DROP COLUMN text;