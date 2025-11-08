import type { VectorStore, VectorSearchResult } from '../../types/vector-store.js';
import type { FalkorDBConnectionManager } from './FalkorDBConnectionManager.js';
import { FalkorDBSchemaManager } from './FalkorDBSchemaManager.js';
import { logger } from '../../utils/logger.js';

export interface FalkorDBVectorStoreOptions {
  connectionManager: FalkorDBConnectionManager;
  indexName?: string;
  dimensions?: number;
  similarityFunction?: 'cosine' | 'euclidean';
  entityNodeLabel?: string;
}

/**
 * FalkorDB implementation of VectorStore interface
 * Uses FalkorDB's native vector search capabilities
 */
export class FalkorDBVectorStore implements VectorStore {
  private readonly connectionManager: FalkorDBConnectionManager;
  private readonly indexName: string;
  private readonly dimensions: number;
  private readonly similarityFunction: 'cosine' | 'euclidean';
  private readonly entityNodeLabel: string;
  private initialized = false;
  private schemaManager: FalkorDBSchemaManager;

  constructor(options: FalkorDBVectorStoreOptions) {
    this.connectionManager = options.connectionManager;
    this.indexName = options.indexName || 'entity_embeddings';
    this.dimensions = options.dimensions || 1536; // Default to OpenAI dimensions
    this.similarityFunction = options.similarityFunction || 'cosine';
    this.entityNodeLabel = options.entityNodeLabel || 'Entity';
    this.schemaManager = new FalkorDBSchemaManager(this.connectionManager);
  }

  /**
   * Initialize the FalkorDB vector store by ensuring the vector index exists
   */
  async initialize(): Promise<void> {
    try {
      // Check if vector index exists - with safety check for tests
      let indexExists = false;
      if (typeof this.schemaManager.vectorIndexExists === 'function') {
        indexExists = await this.schemaManager.vectorIndexExists(this.indexName);
      } else {
        logger.warn(
          'vectorIndexExists method not available on schemaManager - this may be a test environment'
        );
      }

      // Create vector index if it doesn't exist
      if (!indexExists) {
        logger.info(`Creating FalkorDB vector index: ${this.indexName}`);
        if (typeof this.schemaManager.createVectorIndex === 'function') {
          await this.schemaManager.createVectorIndex(
            this.indexName,
            this.entityNodeLabel,
            'embedding',
            this.dimensions,
            this.similarityFunction
          );
        } else {
          logger.warn(
            'createVectorIndex method not available on schemaManager - this may be a test environment'
          );
        }
      } else {
        logger.info(`Using existing FalkorDB vector index: ${this.indexName}`);
      }

      this.initialized = true;
    } catch (error) {
      logger.error('Failed to initialize FalkorDB vector store', error);
      throw error;
    }
  }

  /**
   * Add or update a vector for an entity
   * @param id Entity ID or name
   * @param vector Embedding vector
   * @param metadata Optional metadata to store with the vector
   */
  async addVector(
    id: string | number,
    vector: number[],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    metadata?: Record<string, any>
  ): Promise<void> {
    this.ensureInitialized();

    // Validate vector dimensions
    if (vector.length !== this.dimensions) {
      throw new Error(
        `Invalid vector dimensions: expected ${this.dimensions}, got ${vector.length}`
      );
    }

    try {
      // Determine entity identifier based on whether id is a string or number
      const entityName = typeof id === 'string' ? id : String(id);

      // Build metadata clause if provided
      const metadataClause = metadata
        ? Object.entries(metadata)
            .map(([key]) => `e.${key} = $${key}`)
            .join(', ')
        : '';

      const setClause = metadataClause
        ? `SET e.embedding = $vector, ${metadataClause}`
        : 'SET e.embedding = $vector';

      const params: Record<string, unknown> = {
        name: entityName,
        vector,
        ...metadata,
      };

      await this.connectionManager.executeQuery(
        `MATCH (e:${this.entityNodeLabel} {name: $name}) ${setClause}`,
        params
      );

      logger.debug(`Added/updated vector for entity: ${entityName}`);
    } catch (error) {
      logger.error('Failed to add vector', error);
      throw error;
    }
  }

  /**
   * Search for vectors similar to the query vector
   *
   * **NOTE:** FalkorDB's current vector search implementation has limitations:
   * - `filter`, `hybridSearch`, and `minSimilarity` options are not yet supported
   * - These options will be explicitly rejected with an error if provided
   * - Only `limit` option is currently supported
   *
   * @param queryVector The vector to search for
   * @param options Search options
   * @returns Array of search results with scores
   * @throws Error if unsupported options are provided
   */
  async search(
    queryVector: number[],
    options?: {
      limit?: number;
      filter?: Record<string, any>;
      hybridSearch?: boolean;
      minSimilarity?: number;
    }
  ): Promise<VectorSearchResult[]> {
    this.ensureInitialized();

    // Validate and reject unsupported options
    if (options?.filter) {
      throw new Error(
        'FalkorDB vector search does not currently support the "filter" option. ' +
          'Please use vector search without filters or implement post-filtering in application code.'
      );
    }

    if (options?.hybridSearch) {
      throw new Error(
        'FalkorDB vector search does not currently support the "hybridSearch" option. ' +
          'Please use standard vector search only.'
      );
    }

    if (options?.minSimilarity !== undefined) {
      throw new Error(
        'FalkorDB vector search does not currently support the "minSimilarity" option. ' +
          'Please filter results by similarity score in application code after retrieval.'
      );
    }

    const limit = options?.limit || 10;

    // Validate limit
    if (limit < 1) {
      throw new Error('Limit must be at least 1');
    }

    // Validate vector dimensions
    if (queryVector.length !== this.dimensions) {
      throw new Error(
        `Invalid query vector dimensions: expected ${this.dimensions}, got ${queryVector.length}`
      );
    }

    try {
      const result = await this.connectionManager.executeQuery(
        `CALL db.idx.vector.queryNodes($indexName, $limit, $queryVector) YIELD node, score
         RETURN node.name as name, score`,
        {
          indexName: this.indexName,
          limit,
          queryVector,
        }
      );

      const records = Array.isArray(result.records) ? result.records : [];
      return records.map((record: any) => ({
        id: record.name || record.node?.name,
        similarity: record.score || 0,
        metadata: record.node || {},
      }));
    } catch (error) {
      logger.error('Failed to search vectors', error);
      throw error;
    }
  }

  /**
   * Remove a vector by entity ID (implementation of VectorStore interface method)
   * @param id Entity ID or name
   */
  async removeVector(id: string | number): Promise<void> {
    return this.deleteVector(id);
  }

  /**
   * Delete a vector by entity ID
   * @param id Entity ID or name
   */
  async deleteVector(id: string | number): Promise<void> {
    this.ensureInitialized();

    try {
      const entityName = typeof id === 'string' ? id : String(id);

      await this.connectionManager.executeQuery(
        `MATCH (e:${this.entityNodeLabel} {name: $name}) REMOVE e.embedding`,
        { name: entityName }
      );

      logger.debug(`Deleted vector for entity: ${entityName}`);
    } catch (error) {
      logger.error('Failed to delete vector', error);
      throw error;
    }
  }

  /**
   * Get a vector by entity ID
   * @param id Entity ID or name
   * @returns The vector or null if not found
   */
  async getVector(id: string | number): Promise<number[] | null> {
    this.ensureInitialized();

    try {
      const entityName = typeof id === 'string' ? id : String(id);

      const result = await this.connectionManager.executeQuery(
        `MATCH (e:${this.entityNodeLabel} {name: $name}) RETURN e.embedding as embedding`,
        { name: entityName }
      );

      const records = Array.isArray(result.records) ? result.records : [];
      if (records.length > 0 && records[0]) {
        const record = records[0] as any;
        return record.embedding || null;
      }

      return null;
    } catch (error) {
      logger.error('Failed to get vector', error);
      return null;
    }
  }

  /**
   * Check if the vector store has been initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('Vector store has not been initialized. Call initialize() first.');
    }
  }
}
