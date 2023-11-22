use super::error::OpenAIInnerError;
use super::prompt::completion_to_output;
use super::prompt::stream_to_output;
use async_openai::config::AzureConfig;
use async_openai::types::ChatCompletionRequestMessage;

use async_openai::types::ChatCompletionRequestUserMessageContent;
use llm_chain::options::Opt;
use llm_chain::options::Options;
use llm_chain::options::OptionsCascade;
use llm_chain::output::Output;
use llm_chain::tokens::TokenCollection;
use tiktoken_rs::get_bpe_from_tokenizer;
use tiktoken_rs::tokenizer::get_tokenizer;

use super::prompt::create_chat_completion_request;
use super::prompt::format_chat_messages;
use async_openai::error::OpenAIError;
use llm_chain::prompt::Prompt;

use llm_chain::tokens::PromptTokensError;
use llm_chain::tokens::{Tokenizer, TokenizerError};
use llm_chain::traits;
use llm_chain::traits::{ExecutorCreationError, ExecutorError};

use async_trait::async_trait;
use llm_chain::tokens::TokenCount;

use std::sync::Arc;

/// The `Executor` struct for the ChatGPT model. This executor uses the `async_openai` crate to communicate with the OpenAI API.
#[derive(Clone)]
pub struct Executor {
    /// The client used to communicate with the OpenAI API.
    client: Arc<async_openai::Client<AzureConfig>>,
    /// The per-invocation options for this executor.
    options: Options,
}

impl Default for Executor {
    fn default() -> Self {
        let options = Options::default();
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
        .unwrap_or_else(|_| "https://openai.fdcyun.com".to_string());
        
        let deployment =  std::env::var("AZURE_OPENAI_CHAT_MODEL")
        .unwrap_or_else(|_| "my-openai-model".to_string());

        let version =  std::env::var("AZURE_OPENAI_VERSION")
        .unwrap_or_else(|_| "2023-10-01-preview".to_string());
    
        let config = AzureConfig::new()
        .with_api_base(endpoint)
        .with_deployment_id(deployment)
        .with_api_version(version);

        let client = Arc::new( async_openai::Client::with_config(config));

        Self { client, options }
    }
}

impl Executor {
    /// Creates a new `Executor` with the given client.
    pub fn for_client(client: async_openai::Client<AzureConfig>, options: Options) -> Self {
        use llm_chain::traits::Executor as _;
        let mut exec = Self::new_with_options(options).unwrap();
        exec.client = Arc::new(client);
        exec
    }

    fn get_model_from_invocation_options(&self, opts: &OptionsCascade) -> String {
        let Some(Opt::Model(model)) = 
        opts.get(llm_chain::options::OptDiscriminants::Model) 
        else {
            return "gpt-3.5-turbo".to_string();
        };
        model.to_name()
    }

    fn cascade<'a>(&'a self, opts: Option<&'a Options>) -> OptionsCascade<'a> {
        let mut v: Vec<&'a Options> = vec![&self.options];
        if let Some(o) = opts {
            v.push(o);
        }
        OptionsCascade::from_vec(v)
    }
}

#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub enum Error {
    OpenAIError(#[from] OpenAIError),
}

#[async_trait]
impl traits::Executor for Executor {
    type StepTokenizer<'a> = OpenAITokenizer;
    /// Creates a new `Executor` with the given options.
    fn new_with_options(options: Options) -> Result<Self, ExecutorCreationError> {
        let mut cfg = AzureConfig::new();
        let opts = OptionsCascade::new().with_options(&options);
        if let Some(Opt::ApiKey(api_key)) = opts.get(llm_chain::options::OptDiscriminants::ApiKey) {
            cfg = cfg.with_api_key(api_key)
        }

        if let Some(Opt::AzureDeployment(chat_model) )= opts.get(llm_chain::options::OptDiscriminants::AzureDeployment){
            cfg = cfg.with_deployment_id(chat_model)
        }

        if let Some(Opt::AzureBaseUrl(endpoint)) =  opts.get(llm_chain::options::OptDiscriminants::AzureBaseUrl){
            cfg = cfg.with_api_base(endpoint)
        }

        if  let Some(Opt::AzureApiVersion(version)) = opts.get(llm_chain::options::OptDiscriminants::AzureApiVersion){
            cfg = cfg.with_api_version(version)
        }
        // dbg!(cfg.clone());

        let client = Arc::new(async_openai::Client::with_config(cfg));
        Ok(Self { client, options })
    }

    async fn execute(&self, options: &Options, prompt: &Prompt) -> Result<Output, ExecutorError> {
        let opts = self.cascade(Some(options));
        let client: Arc<async_openai::Client<AzureConfig>> = self.client.clone();
        let model = self.get_model_from_invocation_options(&opts);
        let input = create_chat_completion_request(model, prompt, opts.is_streaming()).unwrap();
        // dbg!(client.clone());
        if opts.is_streaming() {
            let res = async move { client.chat().create_stream(input).await }
                .await
                .map_err(|e| ExecutorError::InnerError(e.into()))?;
            Ok(stream_to_output(res))
        } else {
            let res = async move { client.chat().create(input).await }
                .await
                .map_err(|e| ExecutorError::InnerError(e.into()))?;
            Ok(completion_to_output(res))
        }
    }

    fn tokens_used(
        &self,
        opts: &Options,
        prompt: &Prompt,
    ) -> Result<TokenCount, PromptTokensError> {
        let opts_cas = self.cascade(Some(opts));
        let model = self.get_model_from_invocation_options(&opts_cas);
        let messages = format_chat_messages(prompt.to_chat()).map_err(|e| match e {
            OpenAIInnerError::StringTemplateError(e) => PromptTokensError::PromptFormatFailed(e),
            _ => PromptTokensError::UnableToCompute,
        })?;
        let tokens_used = num_tokens_from_messages(&model, &messages)
            .map_err(|_| PromptTokensError::NotAvailable)?;

        Ok(TokenCount::new(
            self.max_tokens_allowed(opts),
            tokens_used as i32,
        ))
    }
    /// Get the context size from the model or return default context size
    fn max_tokens_allowed(&self, opts: &Options) -> i32 {
        let opts_cas = self.cascade(Some(opts));
        let model = self.get_model_from_invocation_options(&opts_cas);
        tiktoken_rs::model::get_context_size(&model)
            .try_into()
            .unwrap_or(4096)
    }

    fn answer_prefix(&self, _prompt: &Prompt) -> Option<String> {
        None
    }

    fn get_tokenizer(&self, options: &Options) -> Result<OpenAITokenizer, TokenizerError> {
        Ok(OpenAITokenizer::new(self.cascade(Some(options))))
    }
}

fn num_tokens_from_messages(
    model: &str,
    messages: &[ChatCompletionRequestMessage],
) -> Result<usize, PromptTokensError> {
    let tokenizer = get_tokenizer(model).ok_or_else(|| PromptTokensError::NotAvailable)?;
    if tokenizer != tiktoken_rs::tokenizer::Tokenizer::Cl100kBase {
        return Err(PromptTokensError::NotAvailable);
    }
    let bpe = get_bpe_from_tokenizer(tokenizer).map_err(|_| PromptTokensError::NotAvailable)?;

    let (tokens_per_message, tokens_per_name) = if model.starts_with("gpt-3.5") {
        (
            4,  // every message follows <im_start>{role/name}\n{content}<im_end>\n
            -1, // if there's a name, the role is omitted
        )
    } else {
        (3, 1)
    };

    let mut num_tokens: i32 = 0;
    for message in messages {
        let (role, content, name) = match message {
            ChatCompletionRequestMessage::System(x) => (
                x.role.to_string(),
                x.content.to_owned().unwrap_or_default(),
                None,
            ),
            ChatCompletionRequestMessage::User(x) => (
                x.role.to_string(),
                x.content
                    .as_ref()
                    .and_then(|x| match x {
                        ChatCompletionRequestUserMessageContent::Text(x) => Some(x.to_string()),
                        _ => None,
                    })
                    .unwrap_or_default(),
                None,
            ),
            ChatCompletionRequestMessage::Assistant(x) => (
                x.role.to_string(),
                x.content.to_owned().unwrap_or_default(),
                None,
            ),
            ChatCompletionRequestMessage::Tool(x) => (
                x.role.to_string(),
                x.content.to_owned().unwrap_or_default(),
                None,
            ),
            ChatCompletionRequestMessage::Function(x) => (
                x.role.to_string(),
                x.content.to_owned().unwrap_or_default(),
                None,
            ),
        };
        num_tokens += tokens_per_message;
        num_tokens += bpe.encode_with_special_tokens(&role).len() as i32;
        num_tokens += bpe.encode_with_special_tokens(&content).len() as i32;
        if let Some(name) = name {
            num_tokens += bpe.encode_with_special_tokens(name).len() as i32;
            num_tokens += tokens_per_name;
        }
    }
    num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
    Ok(num_tokens as usize)
}

pub struct OpenAITokenizer {
    model_name: String,
}

impl OpenAITokenizer {
    pub fn new(options: OptionsCascade) -> Self {
        let model_name = match options.get(llm_chain::options::OptDiscriminants::Model) {
            Some(Opt::Model(model_name)) => model_name.to_name(),
            _ => "gpt-3.5-turbo".to_string(),
        };
        Self::for_model_name(model_name)
    }
    /// Creates an OpenAITokenizer for the passed in model name
    pub fn for_model_name<S: Into<String>>(model_name: S) -> Self {
        let model_name: String = model_name.into();
        Self { model_name }
    }

    fn get_bpe_from_model(&self) -> Result<tiktoken_rs::CoreBPE, PromptTokensError> {
        use tiktoken_rs::get_bpe_from_model;
        get_bpe_from_model(&self.model_name).map_err(|_| PromptTokensError::NotAvailable)
    }
}

impl Tokenizer for OpenAITokenizer {
    fn tokenize_str(&self, doc: &str) -> Result<TokenCollection, TokenizerError> {
        Ok(self
            .get_bpe_from_model()
            .map_err(|_| TokenizerError::TokenizationError)?
            .encode_ordinary(doc)
            .into())
    }

    fn to_string(&self, tokens: TokenCollection) -> Result<String, TokenizerError> {
        let res = self
            .get_bpe_from_model()
            .map_err(|_e| TokenizerError::ToStringError)?
            .decode(tokens.as_usize()?)
            .map_err(|_e| TokenizerError::ToStringError)?;
        Ok(res)
    }
}
