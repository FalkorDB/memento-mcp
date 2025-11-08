import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FalkorDBConnectionManager } from '../../falkordb/FalkorDBConnectionManager';
import type { FalkorDBConfig } from '../../falkordb/FalkorDBConfig';

// Mock the falkordb client
vi.mock('falkordb', () => {
  const mockQuery = vi.fn().mockResolvedValue({ data: [] });
  const mockDisconnect = vi.fn().mockResolvedValue(undefined);
  const mockConnect = vi.fn().mockResolvedValue(undefined);

  const mockClient = {
    query: mockQuery,
    disconnect: mockDisconnect,
    connect: mockConnect,
  };

  const mockCreateClient = vi.fn().mockReturnValue(mockClient);

  return {
    createClient: mockCreateClient,
  };
});

// Mock @falkordb/graph
vi.mock('@falkordb/graph', () => {
  return {
    Graph: vi.fn().mockImplementation(() => {
      return {
        query: vi.fn().mockResolvedValue({
          data: [],
          headers: [],
        }),
        roQuery: vi.fn().mockResolvedValue({
          data: [],
          headers: [],
        }),
      };
    }),
  };
});

describe('FalkorDBConnectionManager', () => {
  let connectionManager: FalkorDBConnectionManager;
  const defaultConfig: Partial<FalkorDBConfig> = {
    host: 'localhost',
    port: 6379,
    graphName: 'test_graph',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(async () => {
    if (connectionManager) {
      await connectionManager.close();
    }
  });

  it('should create a connection with default config', () => {
    connectionManager = new FalkorDBConnectionManager();
    expect(connectionManager).toBeDefined();
  });

  it('should create a connection with custom config', () => {
    const customConfig: Partial<FalkorDBConfig> = {
      host: 'custom-host',
      port: 6380,
      graphName: 'custom_graph',
    };

    connectionManager = new FalkorDBConnectionManager(customConfig);
    expect(connectionManager).toBeDefined();
  });

  it('should execute a query and return results', async () => {
    connectionManager = new FalkorDBConnectionManager(defaultConfig);
    const mockResult = { records: [] };

    // Execute a query
    const result = await connectionManager.executeQuery('MATCH (n) RETURN n', {});

    expect(result).toBeDefined();
    expect(result.records).toBeDefined();
  });

  it('should get a session', async () => {
    connectionManager = new FalkorDBConnectionManager(defaultConfig);
    const session = await connectionManager.getSession();

    expect(session).toBeDefined();
    expect(session.run).toBeDefined();
    expect(session.close).toBeDefined();
  });

  it('should close the connection', async () => {
    connectionManager = new FalkorDBConnectionManager(defaultConfig);
    await expect(connectionManager.close()).resolves.not.toThrow();
  });
});
