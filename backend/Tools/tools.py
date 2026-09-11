from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import tool, DuckDuckGoSearchRun
import arxiv
import requests
from agent.llm_models import google_llm


@tool
def duckduckgo_search_tool(query: str):
    """
    Search for information on DuckDuckGo based on the provided query.

    Args:
        query (str): The search query.

    """

    if not query:
        raise ValueError(f"Query is needed for this tool to run!")
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
            
    try:
        result = search.invoke(query)
    except Exception as e:
        return f"SOmething went wrong with duck duck go tool: {e}"

    return result

@tool
def arxiv_tool(query: str):
    """
    Search arXiv for research papers relevant to the given query.

    Args:
        query: The topic or keywords to search for.

    Returns:
        A formatted list of the most relevant arXiv papers.
    """
    if not query:
        raise ValueError("Query is needed to run this tool")

    try:
        client = arxiv.Client(
            page_size=3,
            delay_seconds=3,
            num_retries=3
        )

        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        outputs = []

        for result in client.results(search):
            outputs.append(
                f"""
                Title : {result.title}
                URL : {result.entry_id}
                Summary : {result.summary}
                Authors: {", ".join(str(author) for author in result.authors)}
                Published: {result.published}
                """
            )

        if not outputs:
            return "No paper found!"

        return "\n".join(outputs)
    except Exception as e:
        return f"Something went wrong with Arxiv_search_tool : {e}"

@tool
def wikipedia_tool(query: str):
    """
    Search Wikipedia for factual information.
    """

    if not query:
        raise ValueError(f"Query is needed for this tool to run!")

    url = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    params = {
        "q": query,
        "limit": 5
    }

    header = {
        "User-Agent": "Ai_memory_study_agent/1.0"
    }

    try:

        response = requests.get(
            url=url,
            params=params,
            headers=header,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()
        pages = data.get("pages", [])

        results = []


        for page in pages:
            results.append(
                f"""
                Title: {page.get('title', '')}
                Description: {page.get('description', '')}
                Excerpt: {page.get('excerpt', '')}
                """
            )

        return "\n".join(results)
    
    except requests.RequestException as e:
        return f"Wikipedia request failed: {e}"
    
    except Exception as e:
        return f"Something went wrong with wikipedia tool: {e}"




query = "What is the tallest mountain in the world?"

res = wikipedia_tool.invoke(
    {
        "query": query
    }
)

print(res)