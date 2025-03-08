from langchain_nvidia_ai_endpoints import ChatNVIDIA

nvidia_api_key = "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"
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=nvidia_api_key)
result = llm.invoke("tell me about python")
print(result)