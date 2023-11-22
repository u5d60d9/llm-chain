use async_trait::async_trait;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::tools::{Describe, Tool, ToolDescription, ToolError};

pub struct GoogleSearch {
    api_key: String,
    cse_id: String,
}

impl GoogleSearch {
    pub fn new(api_key: String,cse_id: String) -> Self {
        Self { api_key,cse_id }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GoogleSearchInput {
    pub query: String,
}

impl From<&str> for GoogleSearchInput {
    fn from(value: &str) -> Self {
        Self {
            query: value.into(),
        }
    }
}

impl From<String> for GoogleSearchInput {
    fn from(value: String) -> Self {
        Self { query: value }
    }
}

impl Describe for GoogleSearchInput {
    fn describe() -> crate::tools::Format {
        vec![("query", "Search query to find necessary information").into()].into()
    }
}

#[derive(Serialize, Deserialize)]
pub struct GoogleSearchOutput {
    pub result: String,
}

impl From<String> for GoogleSearchOutput {
    fn from(value: String) -> Self {
        Self { result: value }
    }
}

impl From<GoogleSearchOutput> for String {
    fn from(val: GoogleSearchOutput) -> Self {
        val.result
    }
}

impl Describe for GoogleSearchOutput {
    fn describe() -> crate::tools::Format {
        vec![(
            "result",
            "Information retrieved from the internet that should answer your query",
        )
            .into()]
        .into()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GoogleWebPage {
    snippet: String,
}

// #[derive(Debug, Serialize, Deserialize)]
// struct GoogleWebPages {
//     value: Vec<GoogleWebPage>,
// }

#[derive(Debug, Serialize, Deserialize)]
struct GoogleSearchResult {
    items: Vec<GoogleWebPage>,
}

#[derive(Debug, Error)]
pub enum GoogleSearchError {
    #[error("No search results were returned")]
    NoResults,
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Request(#[from] reqwest::Error),
}

impl ToolError for GoogleSearchError {}

#[async_trait]
impl Tool for GoogleSearch {
    type Input = GoogleSearchInput;

    type Output = GoogleSearchOutput;

    type Error = GoogleSearchError;

    async fn invoke_typed(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let response = client
            .request(Method::GET, "https://customsearch.googleapis.com/customsearch/v1")
            .query(&[("q", &input.query),("cx", &self.cse_id.clone())])
            .header("X-goog-api-key", self.api_key.clone())
            .send()
            .await?
            .json::<GoogleSearchResult>()
            .await?;
        let answer = response
            .items
            .first()
            .ok_or(GoogleSearchError::NoResults)?
            .snippet
            .clone();
        Ok(answer.into())
    }

    fn description(&self) -> ToolDescription {
        ToolDescription::new(
            "Google search",
            "Useful for when you need to answer questions about current events. Input should be a search query.",
            "Use this to get information about current events.",
            GoogleSearchInput::describe(),
            GoogleSearchOutput::describe(),
        )
    }
}
