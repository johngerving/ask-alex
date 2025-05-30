-- Persist the IDs as real columns.
ALTER TABLE documents ADD COLUMN id text GENERATED ALWAYS AS (document->>'id_') STORED;
ALTER TABLE data_llamaindex_docs ADD COLUMN document_id text GENERATED ALWAYS AS (metadata_->>'document_id') STORED;

-- Index the new columns.
CREATE UNIQUE INDEX IF NOT EXISTS documents_id_idx ON documents(id);
CREATE INDEX IF NOT EXISTS data_llamaindex_docs_doc_id_idx ON data_llamaindex_docs(document_id);

-- Refresh stats on the new columns.
ANALYZE documents;
ANALYZE data_llamaindex_docs;

---- create above / drop below ----

DROP INDEX documents_id_idx;
DROP INDEX data_llamaindex_docs_doc_id_idx;
ALTER TABLE documents DROP COLUMN id;
ALTER TABLE data_llamaindex_docs DROP COLUMN document_id;