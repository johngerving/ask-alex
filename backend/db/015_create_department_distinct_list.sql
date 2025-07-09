CREATE TABLE departments_distinct (
    department text PRIMARY KEY
);

INSERT INTO departments_distinct (department)
SELECT DISTINCT JSONB_ARRAY_ELEMENTS(document->'metadata'->'department') FROM documents 
ON CONFLICT DO NOTHING;

-- Maintenance trigger
CREATE OR REPLACE FUNCTION upd_departments_distinct() RETURNS trigger AS $$
BEGIN
    IF jsonb_typeof(NEW.document->'metadata'->'department') = 'array' THEN
        INSERT INTO departments_distinct (department)
        SELECT jsonb_array_elements_text(NEW.document->'metadata'->'department')
        ON CONFLICT DO NOTHING;   -- ignore duplicates
    END IF;

    RETURN NEW;   -- value ignored for AFTER triggers
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_departments_distinct 
AFTER INSERT OR UPDATE OF document ON documents
FOR EACH ROW EXECUTE FUNCTION upd_departments_distinct();

---- create above / drop below ----

DROP TRIGGER trg_departments_distinct ON documents;
DROP FUNCTION upd_departments_distinct();
DROP TABLE departments_distinct;
