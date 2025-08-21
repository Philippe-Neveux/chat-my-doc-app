#!/usr/bin/env python3
"""
Demo script for Gradio App with RAG Integration

This script demonstrates the integrated Gradio app with both direct chat
and RAG-enhanced chat functionality using IMDB movie reviews.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chat_my_doc_app.app import create_chat_interface
from chat_my_doc_app.chats import get_available_models, test_rag_connection
from chat_my_doc_app.config import get_config


def demo_app_integration():
    """Demonstrate the integrated Gradio app with RAG functionality."""

    print("🎬 Chat My Doc App - Gradio + RAG Integration Demo")
    print("=" * 60)

    # Test system components
    print("\n🔧 Testing System Components...")

    # Test configuration
    try:
        config = get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   📊 Qdrant host: {config.get('qdrant', {}).get('host', 'Not set')}")
        print(f"   🧮 LLM API: {config.get('llm', {}).get('api_url', 'Not set')}")
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

    # Test available models
    try:
        models = get_available_models()
        print(f"✅ Available models: {', '.join(models)}")
    except Exception as e:
        print(f"❌ Models error: {str(e)}")
        return False

    # Test RAG connection
    try:
        rag_connected = test_rag_connection()
        if rag_connected:
            print(f"✅ RAG system connected and working")
        else:
            print(f"⚠️  RAG system connection issues (see logs)")
    except Exception as e:
        print(f"❌ RAG system error: {str(e)}")
        print(f"   🔧 This might be due to Qdrant not being available")
        print(f"   📝 The app will still work in direct chat mode")

    print(f"\n🚀 Creating Gradio Interface...")

    try:
        # Create the interface (without launching)
        interface = create_chat_interface()
        print(f"✅ Gradio interface created successfully!")

        print(f"\n🎯 Integration Features:")
        print(f"   💬 Direct Chat: Standard conversation with Gemini models")
        print(f"   🎬 RAG Chat: Movie review-enhanced responses using IMDB data")
        print(f"   🔄 Mode Toggle: Switch between direct and RAG modes")
        print(f"   🎮 Model Selection: Choose from {len(models)} available models")
        print(f"   📝 Custom System Prompts: Customize AI behavior")
        print(f"   🧠 Conversation Memory: Maintains context across messages")

        print(f"\n📋 How to Use:")
        print(f"   1. Run: python -m chat_my_doc_app.app")
        print(f"   2. Toggle 'Enable RAG' for movie review context")
        print(f"   3. Ask questions like:")
        print(f"      • 'What are some highly rated action movies?'")
        print(f"      • 'Tell me about romantic comedies from the 2000s'")
        print(f"      • 'Which movies have the best reviews?'")
        print(f"   4. Disable RAG for general conversation")

        return True

    except Exception as e:
        print(f"❌ Interface creation failed: {str(e)}")
        return False


def show_integration_architecture():
    """Show the integration architecture diagram."""

    print("\n🏗️ Integration Architecture:")
    print("─" * 50)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        GRADIO WEB INTERFACE                        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                    [User Toggle]
                          │
            ┌─────────────┴─────────────┐
            │                           │
   ┌────────▼────────┐       ┌─────────▼─────────┐
   │  DIRECT CHAT    │       │    RAG CHAT       │
   │                 │       │                   │
   │ • LangGraph     │       │ • RAG Workflow    │
   │ • Gemini LLM    │       │ • Movie Context   │
   │ • Memory        │       │ • Citations       │
   └─────────────────┘       └───────────┬───────┘
                                         │
                              ┌─────────▼─────────┐
                              │   RAG WORKFLOW    │
                              │                   │
                              │ ┌─────┐ ┌─────┐   │
                              │ │RETV │→│ GEN │   │
                              │ └──┬──┘ └──┬──┘   │
                              │    │       │      │
                              │ ┌──▼────┐  │      │
                              │ │RESPOND│◄─┘      │
                              │ └───────┘         │
                              └─────────┬─────────┘
                                        │
                           ┌────────────▼────────────┐
                           │     QDRANT VECTOR DB    │
                           │   (IMDB Movie Reviews)  │
                           └─────────────────────────┘

Features:
• Seamless mode switching (Direct ↔ RAG)
• Conversation memory in both modes
• Rich movie review context with citations
• Stream-based responses for better UX
• Model selection (Gemini 2.0 Flash, Pro, etc.)
• Custom system prompts
""")


if __name__ == "__main__":
    print("🎬 Chat My Doc App - Integration Demo")

    show_integration_architecture()

    success = demo_app_integration()

    if success:
        print(f"\n✨ Integration demo completed successfully!")
        print(f"🚀 Ready to launch the application!")
        print(f"\nNext steps:")
        print(f"  1. Ensure Qdrant is running with IMDB data")
        print(f"  2. Set CLOUD_RUN_API_URL environment variable")
        print(f"  3. Launch: python -m chat_my_doc_app.app")
    else:
        print(f"\n❌ Integration demo found issues.")
        print(f"Please check the errors above and fix configuration.")
        sys.exit(1)
