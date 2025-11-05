import type { FalkorDBConnectionManager } from './FalkorDBConnectionManager.js';
import { DEFAULT_FALKORDB_CONFIG, type FalkorDBConfig } from './FalkorDBConfig.js';
import { logger } from '../../utils/logger.js';

/**
 * Manages FalkorDB schema operations like creating constraints and indexes
 */
export class FalkorDBSchemaManager {
  private connectionManager: FalkorDBConnectionManager;
  private config: FalkorDBConfig;
  private debug: boolean;

  /**
   * Creates a new FalkorDB schema manager
   * @param connectionManager A FalkorDB connection manager instance
   * @param config FalkorDB configuration (optional)
   * @param debug Whether to enable debug logging (defaults to true)
   */
  constructor(
    connectionManager: FalkorDBConnectionManager,
    config?: Partial<FalkorDBConfig>,
    debug = true
  ) {
    this.connectionManager = connectionManager;
    this.config = {
      ...DEFAULT_FALKORDB_CONFIG,
      ...config,
    };
    this.debug = debug;
  }

  /**
   * Log debug messages if debug mode is enabled
   * @param message Debug message to log
   */
  private log(message: string): void {
    if (this.debug) {
      logger.debug(`[FalkorDBSchemaManager] ${message}`);
    }
  }

  /**
   * Lists all constraints in the database
   * @returns Array of constraint information
   */
  async listConstraints(): Promise<Record<string, unknown>[]> {
    this.log('Listing existing constraints...');
    try {
      const result = await this.connectionManager.executeQuery('SHOW CONSTRAINTS');
      const constraints = Array.isArray(result.records) ? result.records : [];
      this.log(`Found ${constraints.length} constraints`);
      return constraints as Record<string, unknown>[];
    } catch (error) {
      this.log(`Error listing constraints: ${error}`);
      return [];
    }
  }

  /**
   * Lists all indexes in the database
   * @returns Array of index information
   */
  async listIndexes(): Promise<Record<string, unknown>[]> {
    this.log('Listing existing indexes...');
    try {
      const result = await this.connectionManager.executeQuery('SHOW INDEXES');
      const indexes = Array.isArray(result.records) ? result.records : [];
      this.log(`Found ${indexes.length} indexes`);
      return indexes as Record<string, unknown>[];
    } catch (error) {
      this.log(`Error listing indexes: ${error}`);
      return [];
    }
  }

  /**
   * Drops a constraint if it exists
   * @param name Name of the constraint to drop
   */
  async dropConstraintIfExists(name: string): Promise<boolean> {
    this.log(`Dropping constraint ${name} if it exists...`);
    try {
      await this.connectionManager.executeQuery(`DROP CONSTRAINT ${name} IF EXISTS`);
      this.log(`Constraint ${name} dropped or didn't exist`);
      return true;
    } catch (error) {
      this.log(`Error dropping constraint ${name}: ${error}`);
      return false;
    }
  }

  /**
   * Drops an index if it exists
   * @param name Name of the index to drop
   */
  async dropIndexIfExists(name: string): Promise<boolean> {
    this.log(`Dropping index ${name} if it exists...`);
    try {
      await this.connectionManager.executeQuery(`DROP INDEX ${name} IF EXISTS`);
      this.log(`Index ${name} dropped or didn't exist`);
      return true;
    } catch (error) {
      this.log(`Error dropping index ${name}: ${error}`);
      return false;
    }
  }

  /**
   * Creates a unique constraint on entity names
   * @param recreate Whether to drop and recreate the constraint if it exists
   */
  async createEntityNameConstraint(recreate = false): Promise<void> {
    const constraintName = 'entity_name_unique';
    this.log(`Creating entity name uniqueness constraint: ${constraintName}`);

    if (recreate) {
      await this.dropConstraintIfExists(constraintName);
    }

    try {
      await this.connectionManager.executeQuery(
        `CREATE CONSTRAINT ${constraintName} IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE`
      );
      this.log('Entity name constraint created successfully');
    } catch (error) {
      this.log(`Error creating entity name constraint: ${error}`);
      throw error;
    }
  }

  /**
   * Creates an index on entity types
   * @param recreate Whether to drop and recreate the index if it exists
   */
  async createEntityTypeIndex(recreate = false): Promise<void> {
    const indexName = 'entity_type_index';
    this.log(`Creating entity type index: ${indexName}`);

    if (recreate) {
      await this.dropIndexIfExists(indexName);
    }

    try {
      await this.connectionManager.executeQuery(
        `CREATE INDEX ${indexName} IF NOT EXISTS FOR (e:Entity) ON (e.entityType)`
      );
      this.log('Entity type index created successfully');
    } catch (error) {
      this.log(`Error creating entity type index: ${error}`);
      throw error;
    }
  }

  /**
   * Creates temporal indexes for efficient time-based queries
   * @param recreate Whether to drop and recreate the indexes if they exist
   */
  async createTemporalIndexes(recreate = false): Promise<void> {
    const indexes = [
      { name: 'entity_validFrom_index', query: 'FOR (e:Entity) ON (e.validFrom)' },
      { name: 'entity_validTo_index', query: 'FOR (e:Entity) ON (e.validTo)' },
    ];

    for (const { name, query } of indexes) {
      this.log(`Creating temporal index: ${name}`);
      if (recreate) {
        await this.dropIndexIfExists(name);
      }

      try {
        await this.connectionManager.executeQuery(`CREATE INDEX ${name} IF NOT EXISTS ${query}`);
        this.log(`Temporal index ${name} created successfully`);
      } catch (error) {
        this.log(`Error creating temporal index ${name}: ${error}`);
        throw error;
      }
    }
  }

  /**
   * Creates a vector index for semantic search
   * @param indexName Name of the vector index
   * @param nodeLabel Label of the nodes to index
   * @param propertyName Property name containing the vector
   * @param dimensions Vector dimensions
   * @param similarityFunction Similarity function to use
   */
  async createVectorIndex(
    indexName: string,
    nodeLabel: string,
    propertyName: string,
    dimensions: number,
    similarityFunction: 'cosine' | 'euclidean' = 'cosine'
  ): Promise<void> {
    this.log(`Creating vector index: ${indexName}`);
    try {
      // FalkorDB vector index syntax
      await this.connectionManager.executeQuery(
        `CREATE VECTOR INDEX ${indexName} IF NOT EXISTS FOR (n:${nodeLabel}) ON (n.${propertyName}) OPTIONS {dimension: ${dimensions}, similarityFunction: '${similarityFunction}'}`
      );
      this.log(`Vector index ${indexName} created successfully`);
    } catch (error) {
      this.log(`Error creating vector index ${indexName}: ${error}`);
      throw error;
    }
  }

  /**
   * Checks if a vector index exists
   * @param indexName Name of the vector index
   * @returns True if the index exists, false otherwise
   */
  async vectorIndexExists(indexName: string): Promise<boolean> {
    try {
      const indexes = await this.listIndexes();
      return indexes.some((index) => index.name === indexName || index.indexName === indexName);
    } catch (error) {
      this.log(`Error checking if vector index exists: ${error}`);
      return false;
    }
  }

  /**
   * Initialize the schema with all required constraints and indexes
   * @param recreate Whether to drop and recreate existing schema elements
   */
  async initializeSchema(recreate = false): Promise<void> {
    this.log('Initializing FalkorDB schema...');

    try {
      // Create entity name constraint
      await this.createEntityNameConstraint(recreate);

      // Create entity type index
      await this.createEntityTypeIndex(recreate);

      // Create temporal indexes
      await this.createTemporalIndexes(recreate);

      this.log('FalkorDB schema initialized successfully');
    } catch (error) {
      this.log(`Error initializing schema: ${error}`);
      throw error;
    }
  }
}
