use llm_chain::{executor, parameters, prompt, step::Step, options};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new ChatGPT executor
    let opts = options!(
        AzureDeployment: "my-openai-model",
        AzureBaseUrl: "https://openai.fdcyun.com",
        AzureApiVersion: "2023-10-01-preview"
    );

    let exec = executor!(azuregpt,opts)?;    // Create our step containing our prompt template
    let step = Step::for_prompt_template(prompt!(
        "You are a bot for making personalized greetings",
        "Make a personalized greeting tweet for {{text}}" // Text is the default parameter name, but you can use whatever you want
    ));

    // A greeting for emil!
    let res = step.run(&parameters!("Emil"), &exec).await?;
    println!("{}", res.to_immediate().await?.as_content());

    // A greeting for you
    let res = step.run(&parameters!("Your Name Here"), &exec).await?;

    println!("{}", res.to_immediate().await?.as_content());

    Ok(())
}
