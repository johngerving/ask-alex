CREATE INDEX documents_metadata_title_idx ON documents USING gin(to_tsvector('english', document->'metadata'->>'title'));

---- create above / drop below ----

DROP INDEX documents_metadata_title_idx;