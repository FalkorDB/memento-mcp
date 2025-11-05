/**
 * Configuration options for FalkorDB
 */
export interface FalkorDBConfig {
  /**
   * The FalkorDB server host (e.g., 'localhost')
   */
  host: string;

  /**
   * The FalkorDB server port
   */
  port: number;

  /**
   * Username for authentication (optional)
   */
  username?: string;

  /**
   * Password for authentication (optional)
   */
  password?: string;

  /**
   * FalkorDB graph name
   */
  graphName: string;

  /**
   * Name of the vector index
   */
  vectorIndexName: string;

  /**
   * Dimensions for vector embeddings
   */
  vectorDimensions: number;

  /**
   * Similarity function to use for vector search
   */
  similarityFunction: 'cosine' | 'euclidean';
}

/**
 * Default FalkorDB configuration
 */
export const DEFAULT_FALKORDB_CONFIG: FalkorDBConfig = {
  host: 'localhost',
  port: 6379,
  username: undefined,
  password: undefined,
  graphName: 'memento',
  vectorIndexName: 'entity_embeddings',
  vectorDimensions: 1536,
  similarityFunction: 'cosine',
};
