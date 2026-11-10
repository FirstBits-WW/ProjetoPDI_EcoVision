from psycopg2 import sql


def insert_rows(cursor, table_name, rows, returning_column=None):
    if not rows:
        print(f"{table_name}: nenhuma linha configurada.")
        return []

    returned_values = []

    for row in rows:
        columns = list(row.keys())
        values = [row[column] for column in columns]

        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders})").format(
            table=sql.Identifier(table_name),
            columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )

        if returning_column:
            query += sql.SQL(" RETURNING {column}").format(
                column=sql.Identifier(returning_column)
            )

        cursor.execute(query, values)

        if returning_column:
            returned_values.append(cursor.fetchone()[0])

    print(f"{table_name}: {len(rows)} linha(s) inserida(s).")
    return returned_values


def fetch_one_id(cursor, query, params, label):
    cursor.execute(query, params)
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Nao foi encontrado registro para: {label}")
    return result[0]


def fetch_unique_id(cursor, query, params, label, ambiguity_hint=None):
    cursor.execute(query, params)
    results = cursor.fetchall()

    if not results:
        raise ValueError(f"Nao foi encontrado registro para: {label}")

    if len(results) > 1:
        hint = f" {ambiguity_hint}" if ambiguity_hint else ""
        raise ValueError(f"Foi encontrado mais de um registro para: {label}.{hint}")

    return results[0][0]
