import asyncio
from app.main import run_pipeline, PipelineRequest

async def main():
    request = PipelineRequest(
        dataset_path="datasets/test.csv", # need a dummy file
        user_intent="test",
        provider="openrouter"
    )
    
    # Create a dummy dataset
    with open("datasets/test.csv", "w") as f:
        f.write("A,B\n1,2\n3,4")
        
    try:
        res = await run_pipeline(request)
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
