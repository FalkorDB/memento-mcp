/**
 * @vitest-environment node
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FalkorDBStorageProvider } from '../../falkordb/FalkorDBStorageProvider';
import { FalkorDBConnectionManager } from '../../falkordb/FalkorDBConnectionManager';
import type { FalkorDBConfig } from '../../falkordb/FalkorDBConfig';
import { KnowledgeGraph, Entity } from '../../../KnowledgeGraphManager';
import { Relation } from '../../../types/relation';

// Mock the FalkorDBConnectionManager
vi.mock('../../falkordb/FalkorDBConnectionManager', () => {
  return {
    FalkorDBConnectionManager: vi.fn().mockImplementation(() => {
      return {
        getSession: vi.fn().mockResolvedValue({
          beginTransaction: vi.fn().mockReturnValue({
            run: vi.fn().mockResolvedValue({
              records: [
                {
                  get: vi.fn().mockImplementation((key) => {
                    if (key === 'e') {
                      return {
                        properties: {
                          id: 'test-id',
                          name: 'test-entity',
                          entityType: 'test',
                          observations: JSON.stringify(['test observation']),
                          version: 1,
                          createdAt: 1234567890,
                          updatedAt: 1234567890,
                          validFrom: 1234567890,
                          validTo: null,
                        },
                      };
                    }
                    return null;
                  }),
                },
              ],
            }),
            commit: vi.fn().mockResolvedValue(undefined),
            rollback: vi.fn().mockResolvedValue(undefined),
          }),
          run: vi.fn().mockResolvedValue({
            records: [],
          }),
          close: vi.fn().mockResolvedValue(undefined),
        }),
        executeQuery: vi.fn().mockResolvedValue({
          records: [],
        }),
        close: vi.fn().mockResolvedValue(undefined),
      };
    }),
  };
});

// Mock the embedding service
vi.mock('../../../embeddings/EmbeddingServiceFactory', () => {
  return {
    EmbeddingServiceFactory: {
      createFromEnvironment: vi.fn().mockReturnValue({
        generateEmbedding: vi.fn().mockResolvedValue(new Array(1536).fill(0.1)),
        getProviderInfo: vi.fn().mockReturnValue({
          provider: 'test',
          model: 'test-model',
          dimensions: 1536,
        }),
      }),
    },
  };
});

describe('FalkorDBStorageProvider', () => {
  let storageProvider: FalkorDBStorageProvider;
  const testConfig: Partial<FalkorDBConfig> = {
    host: 'localhost',
    port: 6379,
    graphName: 'test_graph',
    vectorIndexName: 'entity_embeddings',
    vectorDimensions: 1536,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    storageProvider = new FalkorDBStorageProvider({
      config: testConfig,
    });
  });

  afterEach(async () => {
    if (storageProvider) {
      await storageProvider.close();
    }
  });

  it('should create a storage provider', () => {
    expect(storageProvider).toBeDefined();
  });

  // Note: StorageProvider doesn't have an initialize() method, schema is initialized in constructor

  it('should save a knowledge graph', async () => {
    const graph: KnowledgeGraph = {
      entities: [
        {
          name: 'test-entity',
          entityType: 'test',
          observations: ['test observation'],
        },
      ],
      relations: [],
    };

    await expect(storageProvider.saveGraph(graph)).resolves.not.toThrow();
  });

  it('should validate graph structure before saving', async () => {
    const invalidGraph = {
      entities: null,
      relations: [],
    } as any;

    await expect(storageProvider.saveGraph(invalidGraph)).rejects.toThrow(
      'Invalid graph: entities and relations must be arrays'
    );
  });

  it('should validate entity names are unique', async () => {
    const graph: KnowledgeGraph = {
      entities: [
        {
          name: 'duplicate',
          entityType: 'test',
          observations: [],
        },
        {
          name: 'duplicate',
          entityType: 'test',
          observations: [],
        },
      ],
      relations: [],
    };

    await expect(storageProvider.saveGraph(graph)).rejects.toThrow('Duplicate entity name');
  });

  it('should create entities with validation', async () => {
    const entities = [
      {
        name: 'test-entity',
        entityType: 'test',
        observations: ['test observation'],
      },
    ];

    await expect(storageProvider.createEntities(entities)).resolves.toBeDefined();
  });

  it('should reject entities without names', async () => {
    const entities = [
      {
        name: '',
        entityType: 'test',
        observations: [],
      },
    ];

    await expect(storageProvider.createEntities(entities)).rejects.toThrow(
      'All entities must have a name'
    );
  });

  it('should create relations with validation', async () => {
    // Update mock to return valid entity nodes for the check query
    const mockConnectionManager = storageProvider['connectionManager'];
    const mockSession = await mockConnectionManager.getSession();
    const mockTx = mockSession.beginTransaction();

    // Mock the check query to return entities exist
    (mockTx.run as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      records: [
        {
          get: vi.fn().mockReturnValue({
            properties: { name: 'entity1' },
          }),
        },
      ],
    });

    // Mock the create relation query
    (mockTx.run as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      records: [
        {
          get: vi.fn().mockImplementation((key) => {
            if (key === 'r') {
              return {
                properties: {
                  id: 'rel-id',
                  relationType: 'test-relation',
                  version: 1,
                  createdAt: Date.now(),
                  updatedAt: Date.now(),
                },
              };
            }
            if (key === 'from') {
              return { properties: { name: 'entity1' } };
            }
            if (key === 'to') {
              return { properties: { name: 'entity2' } };
            }
            return null;
          }),
        },
      ],
    });

    const relations: Relation[] = [
      {
        from: 'entity1',
        to: 'entity2',
        relationType: 'test-relation',
      },
    ];

    await expect(storageProvider.createRelations(relations)).resolves.toBeDefined();
  });

  it('should reject relations without from field', async () => {
    const relations = [
      {
        from: '',
        to: 'entity2',
        relationType: 'test-relation',
      },
    ] as Relation[];

    await expect(storageProvider.createRelations(relations)).rejects.toThrow(
      'All relations must have a from field'
    );
  });

  it('should load the knowledge graph', async () => {
    const graph = await storageProvider.loadGraph();

    expect(graph).toBeDefined();
    expect(graph.entities).toBeDefined();
    expect(graph.relations).toBeDefined();
  });

  it('should search nodes by query', async () => {
    const results = await storageProvider.searchNodes('test', { limit: 10 });

    expect(results).toBeDefined();
    expect(results.entities).toBeDefined();
  });

  it('should close the provider', async () => {
    await expect(storageProvider.close()).resolves.not.toThrow();
  });
});
