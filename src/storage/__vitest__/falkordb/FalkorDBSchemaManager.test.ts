import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FalkorDBSchemaManager } from '../../falkordb/FalkorDBSchemaManager';
import { FalkorDBConnectionManager } from '../../falkordb/FalkorDBConnectionManager';
import type { FalkorDBConfig } from '../../falkordb/FalkorDBConfig';

// Mock the FalkorDBConnectionManager
vi.mock('../../falkordb/FalkorDBConnectionManager', () => {
  const mockExecuteQuery = vi.fn().mockResolvedValue({ records: [] });
  return {
    FalkorDBConnectionManager: vi.fn().mockImplementation(() => ({
      executeQuery: mockExecuteQuery,
      close: vi.fn().mockResolvedValue(undefined),
    })),
  };
});

describe('FalkorDBSchemaManager', () => {
  let schemaManager: FalkorDBSchemaManager;
  let connectionManager: FalkorDBConnectionManager;
  const testConfig: Partial<FalkorDBConfig> = {
    host: 'localhost',
    port: 6379,
    graphName: 'test_graph',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    connectionManager = new FalkorDBConnectionManager(testConfig);
    schemaManager = new FalkorDBSchemaManager(connectionManager, testConfig as FalkorDBConfig);
  });

  afterEach(async () => {
    // Cleanup if needed
  });

  it('should create entity name constraint', async () => {
    await schemaManager.createEntityNameConstraint();

    expect(connectionManager.executeQuery).toHaveBeenCalled();
  });

  it('should create a vector index for entity embeddings', async () => {
    await schemaManager.createVectorIndex('entity_embeddings', 'Entity', 'embedding', 1536);

    expect(connectionManager.executeQuery).toHaveBeenCalledWith(
      expect.stringContaining('CREATE VECTOR INDEX entity_embeddings')
    );
  });

  it('should check if a vector index exists', async () => {
    // Mock the toObject method on records
    const mockRecord = {
      toObject: vi.fn().mockReturnValue({ name: 'entity_embeddings' }),
    };

    (connectionManager.executeQuery as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      records: [mockRecord],
    });

    const exists = await schemaManager.vectorIndexExists('entity_embeddings');

    expect(connectionManager.executeQuery).toHaveBeenCalledWith('SHOW INDEXES');
    expect(exists).toBe(true);
  });

  it('should return false when vector index does not exist', async () => {
    (connectionManager.executeQuery as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      records: [],
    });

    const exists = await schemaManager.vectorIndexExists('non_existent_index');

    expect(exists).toBe(false);
  });

  it('should initialize the schema', async () => {
    await schemaManager.initializeSchema();

    // Should call createEntityConstraints and createVectorIndex
    expect(connectionManager.executeQuery).toHaveBeenCalled();
  });

  // Note: FalkorDBSchemaManager doesn't have a close() method
});
