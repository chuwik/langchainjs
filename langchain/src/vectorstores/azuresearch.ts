import {
  AzureKeyCredential,
  SearchIndexClient,
  SearchIndex,
  VectorSearchAlgorithmConfiguration,
  SearchRequestOptions,
} from "@azure/search-documents";

import * as uuid from "uuid";

import { Document } from "../document.js";
import { Embeddings } from "../embeddings/base.js";
import { VectorStore } from "./base.js";

// type SearchType = "similarity" | "hybrid" | "semanticHybrid";

const MAX_UPLOAD_BATCH_SIZE = 1000;

export interface AzureSearchArgs {
  readonly endpoint: string;
  readonly credential: AzureKeyCredential;

  readonly indexName?: string;
  readonly semanticConfigurationName?: string;

  readonly idFieldName?: string;
  readonly contentFieldName?: string;
  readonly contentVectorFieldName?: string;
  readonly metadataFieldName?: string;

  readonly vectorSearchAlgorithmConfig?: VectorSearchAlgorithmConfiguration;
  readonly vectorSearchDimensions?: number;
}

export class AzureSearchVectorStore extends VectorStore {
  declare FilterType: SearchRequestOptions<object>;

  private readonly _client: SearchIndexClient;

  private readonly _indexName: string;

  private readonly _semanticConfigurationName?: string;

  private readonly _idFieldName: string;

  private readonly _contentFieldName: string;

  private readonly _contentVectorFieldName: string;

  private readonly _metadataFieldName: string;

  private readonly _vectorSearchAlgorithmConfig: VectorSearchAlgorithmConfiguration;

  private readonly _vectorSearchDimensions: number;

  constructor(embeddings: Embeddings, args: AzureSearchArgs) {
    super(embeddings, args);

    this._semanticConfigurationName = args.semanticConfigurationName;

    this._idFieldName = args.idFieldName ?? "id";
    this._contentFieldName = args.contentFieldName ?? "content";
    this._contentVectorFieldName =
      args.contentVectorFieldName ?? "content_vector";
    this._metadataFieldName = args.metadataFieldName ?? "metadata";

    this._vectorSearchAlgorithmConfig = args.vectorSearchAlgorithmConfig ?? {
      name: "default",
      kind: "hnsw",
      parameters: {
        m: 4,
        efConstruction: 400,
        efSearch: 500,
        metric: "cosine",
      },
    };

    this._vectorSearchDimensions = args.vectorSearchDimensions ?? 1536;

    this._client = new SearchIndexClient(args.endpoint, args.credential);
    this._indexName = args.indexName || "documents";
  }

  async addDocuments(
    documents: Document[],
    options?: { ids?: string[] }
  ): Promise<string[]> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: { ids?: string[] }
  ): Promise<string[]> {
    await this._ensureIndexExists();

    const ids: string[] = [];
    let data: object[] = [];

    await Promise.all(
      documents.map(async (x, i) => {
        // Use provided key otherwise use default key
        const key = options?.ids ? options.ids[i] : uuid.v4();

        data.push({
          "@search.action": "upload",
          [this._idFieldName]: key,
          [this._contentFieldName]: x.pageContent,
          [this._contentVectorFieldName]: vectors[i],
          [this._metadataFieldName]: JSON.stringify(x.metadata),
        });

        ids.push(key);

        // Upload data in batches
        if (data.length === MAX_UPLOAD_BATCH_SIZE) {
          await this._client
            .getSearchClient(this._indexName)
            .uploadDocuments(data);

          data = [];
        }
      })
    );

    // Considering case where data is an exact multiple of batch-size entries
    if (data.length === 0) {
      return ids;
    }

    await this._client.getSearchClient(this._indexName).uploadDocuments(data);

    return ids;
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"]
  ): Promise<[Document, number][]> {
    const response = await this._client
      .getSearchClient(this._indexName)
      .search(undefined, {
        ...filter,
        vector: {
          value: query,
          kNearestNeighborsCount: k,
          fields: [this._contentVectorFieldName],
        },
      });

    const results: [Document, number][] = [];
    for await (const x of response.results) {
      const doc = x.document as { [key: string]: string };

      results.push([
        new Document({
          pageContent: doc[this._contentFieldName],
          metadata: JSON.parse(doc[this._metadataFieldName]),
        }),
        x.score,
      ]);
    }

    return results;
  }

  async similaritySearch(
    query: string,
    k = 4,
    filter: this["FilterType"] | undefined = undefined
  ): Promise<Document[]> {
    const results = await this.similaritySearchVectorWithScore(
      await this.embeddings.embedQuery(query),
      k,
      filter
    );

    return results.map((result) => result[0]);
  }

  // async similaritySearchWithScore(
  //   query: string,
  //   k = 4,
  //   filter: this["FilterType"] | undefined = undefined
  // ): Promise<[Document, number][]> {
  //   return this.similaritySearchVectorWithScore(
  //     await this.embeddings.embedQuery(query),
  //     k,
  //     filter
  //   );
  // }

  // async similaritySearchVectorWithScore(): Promise<[Document, number][]> {
  //   // if (filter === "similarity") {
  //   //     throw new Error("Filter type similarity not supported by Azure Search");
  //   // } else if (filter === "hybrid") {
  //   //     throw new Error("Filter type similarity not supported by Azure Search");
  //   // } else if (filter === "semanticHybrid") {
  //   //     throw new Error("Filter type similarity not supported by Azure Search");
  //   // }

  //   this._client.getSearchClient(this._indexName).search("query", {  });

  //   throw new Error("Filter type similarity not supported by Azure Search");
  // }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig: AzureSearchArgs
  ): Promise<AzureSearchVectorStore> {
    const docs = texts.map(
      (text, i) =>
        new Document({
          pageContent: text,
          metadata: Array.isArray(metadatas) ? metadatas[i] : metadatas,
        })
    );

    return this.fromDocuments(docs, embeddings, dbConfig);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: AzureSearchArgs
  ): Promise<AzureSearchVectorStore> {
    const instance = new this(embeddings, dbConfig);
    await instance.addDocuments(docs);
    return instance;
  }

  // static fromExistingIndex(): Promise<AzureSearchVectorStore> {
  //   throw new Error("Method not implemented.");
  // }

  private async _ensureIndexExists(): Promise<void> {
    if (await this._client.getIndex(this._indexName)) {
      return;
    }

    const index: SearchIndex = {
      name: this._indexName,
      fields: [
        {
          name: this._idFieldName,
          type: "Edm.String",
          key: true,
          filterable: true,
        },
        {
          name: this._contentFieldName,
          type: "Edm.String",
          searchable: true,
        },
        {
          name: this._contentVectorFieldName,
          type: "Collection(Edm.Single)",
          searchable: true,
          vectorSearchDimensions: this._vectorSearchDimensions,
          vectorSearchConfiguration: this._vectorSearchAlgorithmConfig.name,
        },
        {
          name: this._metadataFieldName,
          type: "Edm.String",
          searchable: true,
        },
      ],
      vectorSearch: {
        algorithmConfigurations: [this._vectorSearchAlgorithmConfig],
      },
    };

    if (this._semanticConfigurationName) {
      index.semanticSettings = {
        configurations: [
          {
            name: this._semanticConfigurationName,
            prioritizedFields: {
              prioritizedContentFields: [{ name: this._contentFieldName }],
            },
          },
        ],
      };
    }

    await this._client.createOrUpdateIndex(index);
  }
}
