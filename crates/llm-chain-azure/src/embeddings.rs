use std::sync::Arc;

use async_openai::{
    config::AzureConfig,
    error::OpenAIError,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
};


use async_trait::async_trait;
use llm_chain::traits::{self, EmbeddingsError};
use thiserror::Error;

pub struct Embeddings {
    client: Arc<async_openai::Client<AzureConfig>>,
    model: String,
}

#[derive(Debug, Error)]
#[error(transparent)]
pub enum OpenAIEmbeddingsError {
    #[error(transparent)]
    Client(#[from] OpenAIError),
    #[error("Request to OpenAI embeddings API was successful but response is empty")]
    EmptyResponse,
}

impl EmbeddingsError for OpenAIEmbeddingsError {}

#[async_trait]
impl traits::Embeddings for Embeddings {
    type Error = OpenAIEmbeddingsError;

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, Self::Error> {
        let req = CreateEmbeddingRequestArgs::default()
            .model(self.model.clone())
            .input(EmbeddingInput::from(texts))
            .build()?;
        self.client
            .embeddings()
            .create(req)
            .await
            .map(|r| r.data.into_iter().map(|e| e.embedding).collect())
            .map_err(|e| e.into())
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>, Self::Error> {
        let req = CreateEmbeddingRequestArgs::default()
            .model(self.model.clone())
            .input(EmbeddingInput::from(query))
            .build()?;
        self.client
            .embeddings()
            .create(req)
            .await
            .map(|r| r.data.into_iter())?
            .map(|e| e.embedding)
            .last()
            .ok_or(OpenAIEmbeddingsError::EmptyResponse)
    }
}

impl Default for Embeddings {
    fn default() -> Self {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
        .unwrap_or_else(|_| "https://openai.fdcyun.com".to_string());
        
        let deployment =  std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "my-openai-model-qianru".to_string());

        let version =  std::env::var("AZURE_OPENAI_VERSION")
        .unwrap_or_else(|_| "2023-08-01-preview".to_string());
    
        let config = AzureConfig::new()
        .with_api_base(endpoint)
        .with_deployment_id(deployment)
        .with_api_version(version);

        let client = Arc::new( async_openai::Client::with_config(config));

        Self {
            client,
            model: "text-embedding-ada-002".to_string(),  //TODO: 如果以后有新的模型。需要解决， model-mapping from azure to openai.
        }
    }
}

impl Embeddings {
    pub fn for_client(client: async_openai::Client<AzureConfig>, model: &str) -> Self {
        Self {
            client: client.into(),
            model: model.to_string(),
        }
    }
}
