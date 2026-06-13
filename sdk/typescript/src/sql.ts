import { HttpClient } from "./http.js";
import type { JsonValue, SqlExecResult, SqlQueryResult } from "./types.js";

export class SqlClient {
  constructor(private readonly http: HttpClient) {}

  query(sql: string, params: JsonValue[] = []): Promise<SqlQueryResult> {
    return this.http.post("/v1/sql/query", { sql, params });
  }

  exec(sql: string, params: JsonValue[] = []): Promise<SqlExecResult> {
    return this.http.post("/v1/sql/exec", { sql, params });
  }
}
