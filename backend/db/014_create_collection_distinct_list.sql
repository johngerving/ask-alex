CREATE TABLE collections_distinct (
    collection text PRIMARY KEY
);

INSERT INTO collections_distinct (collection)
SELECT DISTINCT document->'metadata'->>'collection'
FROM  documents 
ON CONFLICT DO NOTHING;

-- Maintenance trigger
CREATE OR REPLACE FUNCTION upd_collections_distinct() RETURNS trigger AS $$
BEGIN
    INSERT INTO collections_distinct (collection)
    VALUES (NEW.document->'metadata'->>'collection')
    ON CONFLICT DO NOTHING;      -- ignore duplicates
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_collections_distinct
AFTER INSERT OR UPDATE OF document ON documents
FOR EACH ROW EXECUTE FUNCTION upd_collections_distinct();

---- create above / drop below ----

DROP TRIGGER trg_collections_distinct ON documents;
DROP FUNCTION upd_collections_distinct();
DROP TABLE collections_distinct;