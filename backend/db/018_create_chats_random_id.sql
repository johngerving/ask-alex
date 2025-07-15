CREATE OR REPLACE FUNCTION random_string(length integer, url boolean DEFAULT FALSE)
	RETURNS text
	AS $$
DECLARE
	chars text[] := '{2,4,5,6,7,8,9,B,C,D,F,G,H,J,L,M,P,Q,R,S,T,V,W,X,Y,Z}';
	result text := '';
	i integer := 0;
BEGIN
	IF url IS TRUE THEN
		chars := '{2,4,5,6,7,8,9,B,C,D,F,G,H,J,L,M,P,Q,R,S,T,V,W,X,Y,Z,b,c,d,f,g,h,j,m,p,r,s,t,v,w,x,y,z}';
	END IF;
	IF length < 0 THEN
		RAISE EXCEPTION 'Given length cannot be less than 0';
	END IF;
	FOR i IN 1..length LOOP
		result := result || chars[ceil(array_length(chars, 1) * random())];
	END LOOP;
	RETURN result;
END;
$$
LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION unique_random(len int, _table text, _col text, url boolean DEFAULT FALSE)
	RETURNS text
	AS $$
DECLARE
	result text;
	numrows int;
BEGIN
	result = random_string(len, url);
	LOOP
		EXECUTE format('select 1 from %s where %I = %L', _table, _col, result);
		get diagnostics numrows = row_count;
		IF numrows = 0 THEN
			RETURN result;
		END IF;
		result = random_string(len, url);
	END LOOP;
END;
$$
LANGUAGE plpgsql;

ALTER TABLE chats ADD COLUMN random_id text UNIQUE;
ALTER TABLE chats ALTER COLUMN random_id SET DEFAULT unique_random(10, 'chats', 'random_id', TRUE);

---- create above / drop below ----

ALTER TABLE chats DROP COLUMN IF EXISTS random_id;
DROP FUNCTION IF EXISTS random_string(length integer, url boolean);
DROP FUNCTION IF EXISTS unique_random(len int, _table text, _col text, url boolean);