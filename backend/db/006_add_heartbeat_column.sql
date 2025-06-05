ALTER TABLE tasks ADD COLUMN heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT now();

---- create above / drop below ----

ALTER TABLE tasks DROP COLUMN heartbeat_at;