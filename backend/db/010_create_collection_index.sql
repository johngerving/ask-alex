CREATE INDEX documents_metadata_collection_idx ON documents USING gin ((document->'metadata'->'collection'));

---- create above / drop below ----

DROP INDEX documents_metadata_collection_idx;