#!/usr/bin/env node

/**
 * FalkorDB CLI Utility
 *
 * This script provides command-line utilities for managing FalkorDB
 * operations for the Memento MCP project.
 */

import { FalkorDBConnectionManager } from '../storage/falkordb/FalkorDBConnectionManager.js';
import { FalkorDBSchemaManager } from '../storage/falkordb/FalkorDBSchemaManager.js';
import { DEFAULT_FALKORDB_CONFIG, type FalkorDBConfig } from '../storage/falkordb/FalkorDBConfig.js';

// Factory types for dependency injection in testing
export type ConnectionManagerFactory = (config: FalkorDBConfig) => FalkorDBConnectionManager;
export type SchemaManagerFactory = (
  connectionManager: FalkorDBConnectionManager,
  config: FalkorDBConfig
) => FalkorDBSchemaManager;

// Default factories that use the actual implementations
const defaultConnectionManagerFactory: ConnectionManagerFactory = (config) =>
  new FalkorDBConnectionManager(config);
const defaultSchemaManagerFactory: SchemaManagerFactory = (connectionManager, config) =>
  new FalkorDBSchemaManager(connectionManager, config);

/**
 * Parse command line arguments into a FalkorDB configuration object
 *
 * @param argv Command line arguments array
 * @returns Object containing configuration and options
 */
export function parseArgs(argv: string[]): {
  config: FalkorDBConfig;
  options: { debug: boolean; recreate: boolean };
} {
  const config = { ...DEFAULT_FALKORDB_CONFIG };
  // Always enable debug by default - it provides useful information
  const options = { debug: true, recreate: false };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === '--host' && i + 1 < argv.length) {
      config.host = argv[++i];
    } else if (arg === '--port' && i + 1 < argv.length) {
      config.port = parseInt(argv[++i], 10);
    } else if (arg === '--graph' && i + 1 < argv.length) {
      config.graphName = argv[++i];
    } else if (arg === '--vector-index' && i + 1 < argv.length) {
      config.vectorIndexName = argv[++i];
    } else if (arg === '--dimensions' && i + 1 < argv.length) {
      config.vectorDimensions = parseInt(argv[++i], 10);
    } else if (arg === '--similarity' && i + 1 < argv.length) {
      const similarity = argv[++i];
      if (similarity === 'cosine' || similarity === 'euclidean') {
        config.similarityFunction = similarity;
      }
    } else if (arg === '--no-debug') {
      // Option to disable debug if needed
      options.debug = false;
    } else if (arg === '--recreate') {
      options.recreate = true;
    }
  }

  return { config, options };
}

/**
 * Test the connection to FalkorDB
 *
 * @param config FalkorDB configuration
 * @param debug Enable debug mode
 * @param connectionManagerFactory Factory for creating connection managers (for testing)
 * @returns true if connection is successful, false otherwise
 */
export async function testConnection(
  config: FalkorDBConfig,
  debug = true,
  connectionManagerFactory: ConnectionManagerFactory = defaultConnectionManagerFactory
): Promise<boolean> {
  console.log('Testing connection to FalkorDB...');
  console.log(`  Host: ${config.host}`);
  console.log(`  Port: ${config.port}`);
  console.log(`  Graph: ${config.graphName}`);

  const connectionManager = connectionManagerFactory(config);

  try {
    if (debug) {
      console.log('Debug: Opening FalkorDB session');
    }
    const session = await connectionManager.getSession();

    if (debug) {
      console.log('Debug: Running test query: RETURN 1 as value');
    }
    const result = await session.run('RETURN 1 as value', {});

    if (result.records.length > 0) {
      const value = result.records[0].get('value');
      console.log(`✅ Connection successful! Test value: ${value}`);
      await session.close();
      await connectionManager.close();
      return true;
    } else {
      console.error('❌ Connection test failed: No records returned');
      await session.close();
      await connectionManager.close();
      return false;
    }
  } catch (error) {
    console.error('❌ Connection test failed:', error);
    try {
      await connectionManager.close();
    } catch (closeError) {
      // Ignore close errors
    }
    return false;
  }
}

/**
 * Initialize the FalkorDB schema
 *
 * @param config FalkorDB configuration
 * @param debug Enable debug mode
 * @param recreate Recreate existing constraints and indexes
 * @param connectionManagerFactory Factory for creating connection managers (for testing)
 * @param schemaManagerFactory Factory for creating schema managers (for testing)
 * @returns true if initialization is successful, false otherwise
 */
export async function initializeSchema(
  config: FalkorDBConfig,
  debug = true,
  recreate = false,
  connectionManagerFactory: ConnectionManagerFactory = defaultConnectionManagerFactory,
  schemaManagerFactory: SchemaManagerFactory = defaultSchemaManagerFactory
): Promise<boolean> {
  console.log('Initializing FalkorDB schema...');
  console.log(`  Host: ${config.host}`);
  console.log(`  Port: ${config.port}`);
  console.log(`  Graph: ${config.graphName}`);
  console.log(`  Vector Index: ${config.vectorIndexName}`);
  console.log(`  Vector Dimensions: ${config.vectorDimensions}`);
  console.log(`  Similarity Function: ${config.similarityFunction}`);

  const connectionManager = connectionManagerFactory(config);
  const schemaManager = schemaManagerFactory(connectionManager, config);

  try {
    await schemaManager.initializeSchema(recreate);
    console.log('✅ Schema initialized successfully!');
    await connectionManager.close();
    return true;
  } catch (error) {
    console.error('❌ Schema initialization failed:', error);
    try {
      await connectionManager.close();
    } catch (closeError) {
      // Ignore close errors
    }
    return false;
  }
}

/**
 * Main CLI handler
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('FalkorDB CLI Utility for Memento MCP');
    console.log('');
    console.log('Usage:');
    console.log('  npm run falkordb:init [options]  - Initialize FalkorDB schema');
    console.log('  npm run falkordb:test [options]  - Test FalkorDB connection');
    console.log('');
    console.log('Options:');
    console.log('  --host <host>           FalkorDB host (default: localhost)');
    console.log('  --port <port>           FalkorDB port (default: 6379)');
    console.log('  --graph <name>          Graph name (default: memento)');
    console.log('  --vector-index <name>   Vector index name (default: entity_embeddings)');
    console.log('  --dimensions <number>   Vector dimensions (default: 1536)');
    console.log('  --similarity <function> Similarity function: cosine|euclidean (default: cosine)');
    console.log('  --recreate              Drop and recreate existing constraints/indexes');
    console.log('  --no-debug              Disable debug output');
    console.log('');
    console.log('Environment Variables:');
    console.log('  FALKORDB_HOST');
    console.log('  FALKORDB_PORT');
    console.log('  FALKORDB_GRAPH_NAME');
    console.log('  FALKORDB_VECTOR_INDEX');
    console.log('  FALKORDB_VECTOR_DIMENSIONS');
    console.log('  FALKORDB_SIMILARITY_FUNCTION');
    process.exit(0);
  }

  const command = args[0];
  const { config, options } = parseArgs(args.slice(1));

  // Override with environment variables if set
  if (process.env.FALKORDB_HOST) {
    config.host = process.env.FALKORDB_HOST;
  }
  if (process.env.FALKORDB_PORT) {
    config.port = parseInt(process.env.FALKORDB_PORT, 10);
  }
  if (process.env.FALKORDB_GRAPH_NAME) {
    config.graphName = process.env.FALKORDB_GRAPH_NAME;
  }
  if (process.env.FALKORDB_VECTOR_INDEX) {
    config.vectorIndexName = process.env.FALKORDB_VECTOR_INDEX;
  }
  if (process.env.FALKORDB_VECTOR_DIMENSIONS) {
    config.vectorDimensions = parseInt(process.env.FALKORDB_VECTOR_DIMENSIONS, 10);
  }
  if (process.env.FALKORDB_SIMILARITY_FUNCTION) {
    const similarity = process.env.FALKORDB_SIMILARITY_FUNCTION;
    if (similarity === 'cosine' || similarity === 'euclidean') {
      config.similarityFunction = similarity;
    }
  }

  switch (command) {
    case 'test':
      {
        const success = await testConnection(config, options.debug);
        process.exit(success ? 0 : 1);
      }
      break;

    case 'init':
      {
        const success = await initializeSchema(config, options.debug, options.recreate);
        process.exit(success ? 0 : 1);
      }
      break;

    default:
      console.error(`Unknown command: ${command}`);
      console.error('Valid commands are: init, test');
      process.exit(1);
  }
}

// Run the CLI if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}
