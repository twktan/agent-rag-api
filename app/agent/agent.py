import os
import time
import json
from langchain.agents import create_agent
from langchain.tools import tool
from app.rag.retrieve import retrieve
from app.rag.prompt import build_prompt
from app.llm.generator import call_llm
from app.eval.evaluator import evaluate_rag
from app.monitoring.mlflow_logger import RAGMLflowLogger


# Tools
def create_tools(store):

    @tool(
        'rag',
        description=
        'Use this for company-specific questions that require internal knowledge.'
    )
    def rag_tool(query: str) -> dict:
        docs = retrieve(query, store, k=3)
        prompt = build_prompt(query, docs)
        answer = call_llm(prompt, rag=True)
        score = evaluate_rag(query, answer, docs)

        return {
            'query': query,
            'docs': docs,
            'prompt': prompt,
            'answer': answer,
            'score': score,
            'route': 'RAG'
        }

    @tool(
        'direct_answer',
        description=
        'Use this for general knowledge questions not related to the company.')
    def direct_tool(query: str) -> dict:
        answer = call_llm(query, rag=False)

        return {
            'query': query,
            'docs': [],
            'prompt': query,
            'answer': answer,
            'score': None,
            'route': 'DIRECT'
        }

    return [rag_tool, direct_tool]


# Agent class
class RAGAgent:

    def __init__(self, store):

        self.store = store

        self.logger = RAGMLflowLogger(tracking_uri=os.getenv(
            'MLFLOW_TRACKING_URI', '<INSERT_MLFLOW_SERVER_URL>'),
                                      experiment_name='agent-rag-system')

        tools = create_tools(store)

        self.agent = create_agent(model='gpt-4o-mini',
                                  tools=tools,
                                  system_prompt='''
You are a routing assistant.

- If the query is about company-specific knowledge, use the 'rag' tool
- Otherwise, use the 'direct_answer' tool

Rules:
- Always call exactly ONE tool
- Do not answer directly
''')

    # Helper method to assist extraction for MLflow monitoring
    def _extract_tool_output(self, result):
        if 'messages' in result:
            for msg in reversed(result['messages']):
                if hasattr(msg, 'type') and msg.type == 'tool':
                    content = msg.content
                    try:
                        return json.loads(content)
                    except:
                        return {'raw': content}
        return None

    # Main function
    def run(self, query: str):

        start = time.time()

        result = self.agent.invoke(
            {'messages': [{
                'role': 'user',
                'content': query
            }]})

        latency = time.time() - start

        tool_output = self._extract_tool_output(result)

        if tool_output is None:
            raise ValueError(
                'No tool output found.')

        # MLflow logging
        try:
            self.logger.log_rag_event(query=tool_output['query'],
                                      answer=tool_output['answer'],
                                      retrieved_chunks=tool_output['docs'],
                                      prompt=tool_output['prompt'],
                                      model_name='gpt-4o-mini',
                                      latency=latency,
                                      score=tool_output['score'],
                                      metadata={
                                          'route': tool_output['route'],
                                          'retriever': 'faiss',
                                          'top_k': 3
                                      })

        except Exception as e:
            print(f'[MLFLOW LOG FAILED] {e}')

        # Final response
        return {'answer': tool_output['answer'], 'route': tool_output['route']}
