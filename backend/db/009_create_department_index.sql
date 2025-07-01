CREATE INDEX documents_metadata_department_idx ON documents USING gin ((document->'metadata'->'department'));

---- create above / drop below ----

DROP INDEX documents_metadata_department_idx;