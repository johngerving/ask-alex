CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION jsonb_to_text(jsonb)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
SELECT CASE
         WHEN $1 IS NULL                   THEN NULL                   -- nothing there
         WHEN jsonb_typeof($1) = 'array'   THEN
                (SELECT string_agg(value,' ')
                   FROM jsonb_array_elements_text($1) AS t(value))     -- flatten array
         ELSE                                $1::text                  -- scalar → text
       END
$$;

CREATE INDEX documents_metadata_author_search_idx ON documents USING gin (to_tsvector('english', jsonb_to_text(document->'metadata'->'author')));

---- create above / drop below ----

DROP FUNCTION jsonb_to_text(jsonb);
DROP INDEX documents_metadata_author_search_idx;