/**
 * @vitest-environment node
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FalkorDBVectorStore } from '../../falkordb/FalkorDBVectorStore.js';
import { FalkorDBConnectionManager } from '../../falkordb/FalkorDBConnectionManager.js';

// Mock FalkorDBConnectionManager
vi.mock('../../falkordb/FalkorDBConnectionManager.js', () => {
  const mockExecuteQuery = vi.fn().mockResolvedValue({
    records: [
      {
        name: 'test-entity',
        score: 0.95,
      },
    ],
  });

  return {
    FalkorDBConnectionManager: vi.fn().mockImplementation(() => {
      return {
        executeQuery: mockExecuteQuery,
        close: vi.fn().mockResolvedValue(undefined),
      };
    }),
  };
});

describe('FalkorDBVectorStore', () => {
  let vectorStore: FalkorDBVectorStore;
  let mockConnectionManager: FalkorDBConnectionManager;

  beforeEach(() => {
    vi.clearAllMocks();
    mockConnectionManager = new FalkorDBConnectionManager();

    vectorStore = new FalkorDBVectorStore({
      connectionManager: mockConnectionManager,
      indexName: 'test_index',
      dimensions: 1536,
      similarityFunction: 'cosine',
      entityNodeLabel: 'Entity',
    });
  });

  afterEach(async () => {
    // Cleanup
  });

  it('should initialize the vector store', async () => {
    await vectorStore.initialize();
    expect(mockConnectionManager.executeQuery).toHaveBeenCalled();
  });

  it('should add a vector', async () => {
    await vectorStore.initialize();
    const vector = new Array(1536).fill(0.1);
    const metadata = { type: 'test' };

    await expect(
      vectorStore.addVector('test-entity', vector, metadata)
    ).resolves.not.toThrow();
  });

  it('should throw error when adding vector with wrong dimensions', async () => {
    await vectorStore.initialize();
    const vector = new Array(100).fill(0.1); // Wrong dimension

    await expect(vectorStore.addVector('test-entity', vector)).rejects.toThrow(
      'Invalid vector dimensions'
    );
  });

  it('should search for similar vectors', async () => {
    await vectorStore.initialize();
    const queryVector = new Array(1536).fill(0.1);

    const results = await vectorStore.search(queryVector, { limit: 10 });

    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
  });

  it('should reject unsupported filter option', async () => {
    await vectorStore.initialize();
    const queryVector = new Array(1536).fill(0.1);

    await expect(
      vectorStore.search(queryVector, { filter: { type: 'test' } })
    ).rejects.toThrow('does not currently support the "filter" option');
  });

  it('should reject unsupported hybridSearch option', async () => {
    await vectorStore.initialize();
    const queryVector = new Array(1536).fill(0.1);

    await expect(
      vectorStore.search(queryVector, { hybridSearch: true })
    ).rejects.toThrow('does not currently support the "hybridSearch" option');
  });

  it('should reject unsupported minSimilarity option', async () => {
    await vectorStore.initialize();
    const queryVector = new Array(1536).fill(0.1);

    await expect(
      vectorStore.search(queryVector, { minSimilarity: 0.8 })
    ).rejects.toThrow('does not currently support the "minSimilarity" option');
  });

  it('should delete a vector', async () => {
    await vectorStore.initialize();

    await expect(vectorStore.deleteVector('test-entity')).resolves.not.toThrow();
  });

  it('should throw error when not initialized', async () => {
    const queryVector = new Array(1536).fill(0.1);

    await expect(vectorStore.search(queryVector)).rejects.toThrow(
      'Vector store has not been initialized'
    );
  });
});
