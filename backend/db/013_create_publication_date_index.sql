CREATE INDEX documents_metadata_publication_date_idx ON documents USING btree(((document->'metadata'->>'publication_date')::int));

---- create above / drop below ----

DROP INDEX documents_metadata_publication_date_idx;