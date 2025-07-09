DROP INDEX documents_text_idx;
CREATE INDEX documents_text_idx ON documents USING bm25 (id, text) WITH (key_field='id');

---- create above / drop below ----

DROP INDEX documents_text_idx;