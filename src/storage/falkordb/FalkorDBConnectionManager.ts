import FalkorDB from 'falkordb';
import type { Graph } from 'falkordb';
import { DEFAULT_FALKORDB_CONFIG, type FalkorDBConfig } from './FalkorDBConfig.js';

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
  private client: typeof FalkorDB | null = null;
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
      const connectionOptions: {
        socket: { host: string; port: number };
        username?: string;
        password?: string;
      } = {
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

      this.client = await FalkorDB.connect(connectionOptions);
    }
  }

  /**
   * Gets a FalkorDB graph instance for executing queries
   * @returns A FalkorDB graph instance
   */
  async getGraph(): Promise<Graph> {
    await this.ensureConnected();
    if (!this.graph && this.client) {
      this.graph = this.client.selectGraph(this.config.graphName);
    }
    if (!this.graph) {
      throw new Error('Failed to get FalkorDB graph instance');
    }
    return this.graph;
  }

  /**
   * Executes a Cypher query
   * @param query The Cypher query
   * @param parameters Query parameters
   * @returns Query result
   */
  async executeQuery(
    query: string,
    parameters?: Record<string, unknown>
  ): Promise<{ records: unknown[] }> {
    const graph = await this.getGraph();
    const result = await graph.query(query, { params: parameters });
    return { records: result };
  }

  /**
   * Closes the FalkorDB client connection
   */
  async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
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
