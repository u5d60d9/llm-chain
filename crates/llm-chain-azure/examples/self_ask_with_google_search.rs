use llm_chain::{
    agents::self_ask_with_search::{Agent, EarlyStoppingConfig},
    executor,options,
    tools::tools::GoogleSearch,
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let opts = options!(
        AzureDeployment: "my-openai-model-gpt4",
        AzureBaseUrl: "https://openai.fdcyun.com",
        AzureApiVersion: "2023-10-01-preview"
    );

    let executor = executor!(azuregpt,opts).unwrap();    
    let google_api_key = std::env::var("GOOGLE_API_KEY").unwrap();
    let cse = std::env::var("GOOGLE_API_CSE").unwrap();
    let search_tool = GoogleSearch::new(google_api_key,cse);
    let agent = Agent::new(
        executor,
        search_tool,
        EarlyStoppingConfig {
            max_iterations: Some(10),
            max_time_elapsed_seconds: Some(30.0),
        },
    );
    let question = "how to write a echart.js line graph with forecast,show me the code";

    let ar = agent
        .run(question)
        .await;
    if ar.is_ok(){
       let (res, intermediate_steps)  = ar.unwrap();
       println!(
        "Are followup questions needed here: {}",
        agent.build_agent_scratchpad(&intermediate_steps)
        );
        println!(
            "Agent final answer: {}",
            res.return_values.get("output").unwrap()
        );
    }
    else{
        print!("{:?}",ar.err());
    }
    
}
