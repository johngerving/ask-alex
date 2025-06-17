CREATE TABLE http_sessions (
    id BIGSERIAL PRIMARY KEY,
    key BYTEA,
    data JSONB,
    created_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    modified_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_on TIMESTAMPTZ NOT NULL
);

CREATE INDEX http_sessions_expiry_idx ON http_sessions (expires_on);
CREATE INDEX http_sessions_key_idx ON http_sessions (key);

---- create above / drop below ----

DROP TABLE http_sessions;