import { createClient, type RedisClientType } from 'falkordb';
import { Graph } from '@falkordb/graph';
import { DEFAULT_FALKORDB_CONFIG, type FalkorDBConfig } from './FalkorDBConfig.js';

type QueryParam = string | number | boolean | null | QueryParam[];
type QueryParams = Record<string, QueryParam>;

/**
 * Wrapper for FalkorDB result records to provide Neo4j-like API
 */
export class FalkorDBRecord {
  private data: any;

  constructor(data: any) {
    this.data = data;
  }

  /**
   * Get a value by key (Neo4j-compatible method)
   * @param key The key to retrieve
   * @returns The value
   */
  get(key: string | number): any {
    if (typeof key === 'number') {
      return this.data[key];
    }

    // For named access, try to find the key in the data
    if (this.data && typeof this.data === 'object') {
      return this.data[key];
    }

    return this.data;
  }

  /**
   * Convert to plain object
   */
  toObject(): any {
    return this.data;
  }
}

/**
 * Wrapper for FalkorDB graph to provide Neo4j Session-like API
 */
class FalkorDBSession {
  private graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  /**
   * Begin a transaction (FalkorDB doesn't have explicit transactions, so we simulate it)
   * @returns A transaction-like object
   */
  beginTransaction(): FalkorDBTransaction {
    return new FalkorDBTransaction(this.graph);
  }

  /**
   * Close the session (no-op for FalkorDB)
   */
  async close(): Promise<void> {
    // No-op - FalkorDB doesn't have sessions
  }

  /**
   * Run a query
   * @param query The Cypher query
   * @param parameters Query parameters
   * @returns Query result
   */
  async run(
    query: string,
    parameters?: Record<string, unknown>
  ): Promise<{ records: FalkorDBRecord[] }> {
    const result = await this.graph.query(query, { params: parameters as QueryParams });
    const records: FalkorDBRecord[] = [];

    if (result && Array.isArray(result)) {
      for (const row of result) {
        records.push(new FalkorDBRecord(row));
      }
    }

    return { records };
  }
}

/**
 * Wrapper to provide Neo4j Transaction-like API for FalkorDB
 */
class FalkorDBTransaction {
  private graph: Graph;
  private queries: Array<{ query: string; params?: Record<string, unknown> }> = [];

  constructor(graph: Graph) {
    this.graph = graph;
  }

  /**
   * Run a query in the transaction
   * @param query The Cypher query
   * @param parameters Query parameters
   * @returns Query result
   */
  async run(
    query: string,
    parameters?: Record<string, unknown>
  ): Promise<{ records: FalkorDBRecord[] }> {
    // Store the query to be executed on commit
    this.queries.push({ query, params: parameters });

    // Execute immediately (FalkorDB doesn't have real transactions)
    const result = await this.graph.query(query, { params: parameters as QueryParams });
    const records: FalkorDBRecord[] = [];

    if (result && Array.isArray(result)) {
      for (const row of result) {
        records.push(new FalkorDBRecord(row));
      }
    }

    return { records };
  }

  /**
   * Commit the transaction (no-op for FalkorDB)
   */
  async commit(): Promise<void> {
    // No-op - queries are already executed
  }

  /**
   * Rollback the transaction (not supported in FalkorDB)
   */
  async rollback(): Promise<void> {
    // Cannot rollback in FalkorDB
    throw new Error('FalkorDB does not support transaction rollback');
  }
}

/**
 * Options for configuring a FalkorDB connection
 * @deprecated Use FalkorDBConfig instead
 */
export interface FalkorDBConnectionOptions {
  host?: string;
  port?: number;
  username?: string;
  password?: string;
  graphName?: string;
}

/**
 * Manages connections to a FalkorDB database
 */
export class FalkorDBConnectionManager {
  private client: RedisClientType | null = null;
  private graph: Graph | null = null;
  private readonly config: FalkorDBConfig;

  /**
   * Creates a new FalkorDB connection manager
   * @param config Connection configuration
   */
  constructor(config?: Partial<FalkorDBConfig> | FalkorDBConnectionOptions) {
    // Handle deprecated options
    if (config && 'host' in config) {
      this.config = {
        ...DEFAULT_FALKORDB_CONFIG,
        ...config,
      };
    } else {
      this.config = {
        ...DEFAULT_FALKORDB_CONFIG,
        ...config,
      };
    }
  }

  /**
   * Ensures the client is connected
   */
  private async ensureConnected(): Promise<void> {
    if (!this.client) {
      const connectionOptions: any = {
        socket: {
          host: this.config.host,
          port: this.config.port,
        },
      };

      if (this.config.username) {
        connectionOptions.username = this.config.username;
      }

      if (this.config.password) {
        connectionOptions.password = this.config.password;
      }

      this.client = createClient(connectionOptions);
      await this.client.connect();
    }
  }

  /**
   * Gets a FalkorDB graph instance for executing queries
   * @returns A FalkorDB graph instance
   */
  async getGraph(): Promise<Graph> {
    await this.ensureConnected();
    if (!this.graph && this.client) {
      // Create a new Graph instance with the client and graph name
      this.graph = new Graph(this.client as any, this.config.graphName);
    }
    if (!this.graph) {
      throw new Error('Failed to get FalkorDB graph instance');
    }
    return this.graph;
  }

  /**
   * Gets a FalkorDB "session" (actually just returns a wrapper for compatibility)
   * @returns A session-like object
   */
  async getSession(): Promise<FalkorDBSession> {
    const graph = await this.getGraph();
    return new FalkorDBSession(graph);
  }

  /**
   * Executes a Cypher query
   * @param query The Cypher query
   * @param parameters Query parameters
   * @returns Query result with Neo4j-compatible structure
   */
  async executeQuery(
    query: string,
    parameters?: Record<string, unknown>
  ): Promise<{ records: FalkorDBRecord[] }> {
    const graph = await this.getGraph();
    const result = await graph.query(query, { params: parameters as QueryParams });

    // Convert FalkorDB result format to Neo4j-like format
    const records: FalkorDBRecord[] = [];

    if (result && Array.isArray(result)) {
      for (const row of result) {
        records.push(new FalkorDBRecord(row));
      }
    }

    return { records };
  }

  /**
   * Closes the FalkorDB client connection
   */
  async close(): Promise<void> {
    if (this.client) {
      await this.client.disconnect();
      this.client = null;
      this.graph = null;
    }
  }

  /**
   * Get the current configuration
   */
  getConfig(): FalkorDBConfig {
    return this.config;
  }
}
