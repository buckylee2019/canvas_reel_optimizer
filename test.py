import json
import boto3

# Create the Bedrock Runtime client using default credentials (IAM roles in ECS)
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

model_input = {
    "taskType": "TEXT_VIDEO",
    "textToVideoParams": {
        "text": "Closeup of a large seashell in the sand, gentle waves flow around the shell. Camera zoom in."
    },
    "videoGenerationConfig": {
        "durationSeconds": 6,
        "fps": 24,
        "dimension": "1280x720",
        "seed": 0,  # Change the seed to get a different result
    },
}
try:
    # Start the asynchronous video generation job.
    invocation = bedrock_runtime.start_async_invoke(
        modelId="amazon.nova-reel-v1:1",
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": "s3://bedrock-video-generation-us-east-1-jlvyiv"
            }
        }
    )

    # Print the response JSON.
    print("Response:")
    print(json.dumps(invocation, indent=2, default=str))

except Exception as e:
    # Implement error handling here.
    message = e.response["Error"]["Message"]
    print(f"Error: {message}")