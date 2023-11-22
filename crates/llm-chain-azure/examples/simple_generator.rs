use llm_chain::{options,executor, parameters, prompt};

// Declare an async main function
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new ChatGPT executor
    let opts = options!(
        AzureDeployment: "my-openai-model",
        AzureBaseUrl: "https://openai.fdcyun.com",
        AzureApiVersion: "2023-10-01-preview"
    );

    let exec = executor!(azuregpt,opts)?;
    // Create our prompt...
    let res = prompt!(
        "You are a robot assistant for making personalized greetings",
        "Make a personalized greeting for Joe"
    )
    .run(&parameters!(), &exec) // ...and run it
    .await?;
    println!("{}", res);
    Ok(())
}
