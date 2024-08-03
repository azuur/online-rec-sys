CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movies_all (
    id CHAR(9),
    title VARCHAR,
    synopsis VARCHAR,
    tags VARCHAR,
    vec_input VARCHAR,
    vec VECTOR(1536),
    PRIMARY KEY (id)
);

CREATE INDEX ON movies_all USING hnsw (vec vector_cosine_ops);

CREATE TABLE IF NOT EXISTS movies_posters (
    movie_id CHAR(9),
    img_bytes BYTEA,
    FOREIGN KEY (movie_id) REFERENCES movies_all(id)
);

CREATE TABLE IF NOT EXISTS users_current (
    id SERIAL PRIMARY KEY,
    vec VECTOR(1536),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users_current_v1 (
    id SERIAL PRIMARY KEY,
    sklearn_model BYTEA,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS batches_current (
    user_id INTEGER,
    movie_id CHAR(9),
    label BOOL,
    added_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users_current(id),
    FOREIGN KEY (movie_id) REFERENCES movies_all(id)
);

CREATE TABLE IF NOT EXISTS users_history (
    id INTEGER,
    vec VECTOR(1536),
    next_batch_id INTEGER,
    moved_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, moved_at),
    FOREIGN KEY (id) REFERENCES users_current(id)
);

CREATE TABLE IF NOT EXISTS users_history_v1 (
    id INTEGER,
    sklearn_model BYTEA,
    next_batch_id INTEGER,
    moved_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, moved_at),
    FOREIGN KEY (id) REFERENCES users_current(id)
);

CREATE TABLE IF NOT EXISTS batches_history (
    batch_id SERIAL,
    user_id INTEGER,
    movie_id CHAR(9),
    label BOOL,
    added_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users_current(id),
    FOREIGN KEY (movie_id) REFERENCES movies_all(id)
)