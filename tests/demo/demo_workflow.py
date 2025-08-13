#!/usr/bin/env python3
"""
Demo script for RAG LangGraph Workflow

This script demonstrates the complete RAG workflow with LangGraph integration.
Run this to see how the three-node workflow (retrieve → generate → respond) works.
"""

import sys

from chat_my_doc_app.rag import RAGImdb
from chat_my_doc_app.config import get_config


def demo_rag_workflow():
    """Demonstrate RAG LangGraph workflow functionality."""

    print("🎬 RAG LangGraph Workflow Demo - IMDB Movie Reviews")
    print("=" * 60)

    try:
        # Initialize RAG workflow
        print("\n🚀 Initializing RAG LangGraph workflow...")
        workflow = RAGImdb()
        print("✅ RAG workflow initialized successfully!")

        # Show workflow information
        workflow_info = workflow.get_workflow_info()
        print(f"\n📋 Workflow Info:")
        print(f"  🔄 Type: {workflow_info['workflow_type']}")
        print(f"  📦 Nodes: {' → '.join(workflow_info['nodes'])}")
        print(f"  🧮 LLM: {workflow_info['llm']['type']} ({workflow_info['llm']['model_name']})")
        print(f"  📏 Max Context: {workflow_info['rag_service']['max_context_length']} chars")

        # Test queries for the workflow
        test_queries = [
            "What are some highly rated action movies?",
            "Tell me about romantic comedies with good reviews",
            "What makes a movie worth watching according to reviews?",
            "Which movies have the worst reviews and why?"
        ]

        print(f"\n🧪 Testing {len(test_queries)} queries through the workflow...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n" + "─" * 50)
            print(f"🔍 Query {i}: '{query}'")
            print("─" * 50)

            try:
                # Process query through workflow
                result = workflow.process_query(query)

                # Display results
                response = result.get('response', 'No response generated')
                citations = result.get('citations', [])
                metadata = result.get('metadata', {})

                print(f"📤 Response:")
                print(f"   {response[:200]}{'...' if len(response) > 200 else ''}")

                print(f"\n📊 Metadata:")
                print(f"   📚 Documents Retrieved: {metadata.get('retrieved_docs', 0)}")
                print(f"   📄 Context Length: {metadata.get('context_length', 0)} chars")
                print(f"   💬 Response Length: {metadata.get('final_response_length', 0)} chars")
                print(f"   🏷️  Citations Included: {'Yes' if metadata.get('citations_included') else 'No'}")

                if citations:
                    print(f"\n🔗 Top Citations:")
                    for citation in citations[:2]:  # Show top 2
                        movie_title = citation.get('movie_title', 'Unknown')
                        score = citation.get('score', 0.0)
                        year = citation.get('year', '')
                        genre = citation.get('genre', '')

                        citation_info = f"   • {movie_title}"
                        if year:
                            citation_info += f" ({year})"
                        if genre:
                            citation_info += f" - {genre}"
                        citation_info += f" (Score: {score:.3f})"

                        print(citation_info)

                # Check for any errors
                if 'error' in metadata:
                    print(f"   ⚠️ Error: {metadata['error']}")

            except Exception as e:
                print(f"  ❌ Error processing query: {str(e)}")

        print(f"\n🎯 Workflow demo completed!")
        print(f"\nYou can now use the RAG workflow in your applications:")
        print(f"```python")
        print(f"from chat_my_doc_app.workflow import create_rag_workflow")
        print(f"")
        print(f"workflow = create_rag_workflow()")
        print(f"result = workflow.process_query('your query')")
        print(f"print(result['response'])")
        print(f"```")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print(f"This might be because:")
        print(f"  - Qdrant service is not running")
        print(f"  - LLM API is not available")
        print(f"  - Configuration is incorrect")
        print(f"\nPlease check your configuration and try again.")
        return False


def show_workflow_architecture():
    """Show the workflow architecture diagram."""

    print("\n🏗️ Workflow Architecture:")
    print("─" * 40)
    print("""
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   RETRIEVE  │───▶│  GENERATE   │───▶│   RESPOND   │
│             │    │             │    │             │
│ • Query     │    │ • Context   │    │ • Response  │
│ • RAG       │    │ • LLM       │    │ • Citations │
│ • Context   │    │ • Prompt    │    │ • Format    │
│ • Citations │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘

Flow:
1. RETRIEVE: Use RAG service to find relevant movie reviews
2. GENERATE: Create response using LLM with retrieved context
3. RESPOND: Format final response with citations
""")


if __name__ == "__main__":
    print("🎬 Chat My Doc App - RAG LangGraph Workflow Demo")

    show_workflow_architecture()

    success = demo_rag_workflow()

    if success:
        print(f"\n✨ RAG LangGraph workflow is working correctly!")
        print(f"Ready to proceed to Step 4: Gradio integration")
    else:
        print(f"\n❌ Please fix the issues above before proceeding.")
        print(f"\nNote: The workflow requires:")
        print(f"  1. Qdrant service running with IMDB data")
        print(f"  2. LLM API endpoint available")
        print(f"  3. Proper configuration in config.yaml")
        sys.exit(1)
