"""
Etapa 1: Chatbot mínimo con LangGraph
--------------------------------------
Un grafo con un único nodo que llama al modelo.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

# Modelo (sustituir por OpenRouter si no se tiene OpenAI)
modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ----- Alternativa OpenRouter (descomentar si se usa) -----
# modelo = ChatOpenAI(
#     model="google/gemini-2.0-flash-exp:free",
#     temperature=0.7,
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
# )


# TODO 1: Nodo que llama al modelo
def llamar_modelo(state: MessagesState) -> dict:
    """Recibe el estado de mensajes y devuelve la respuesta del modelo."""
    respuesta = modelo.invoke(state["messages"])
    # Devolvemos la respuesta envuelta en una lista; gracias al reducer
    # add_messages de MessagesState, se concatenará al historial existente
    # en lugar de sobreescribirlo.
    return {"messages": [respuesta]}


# TODO 2: Construir el grafo
builder = StateGraph(MessagesState)
builder.add_node("chatbot", llamar_modelo)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# TODO 3: Compilar el grafo
graph = builder.compile()

# TODO 4: Visualizar el grafo en ASCII
print(graph.get_graph().draw_ascii())

# TODO 5: Invocar el grafo
resultado = graph.invoke(
    {"messages": [{"role": "user", "content": "¿Qué es Python?"}]}
)

# Imprimir la respuesta del modelo (último mensaje del historial)
print("\n--- Respuesta del modelo ---")
print(resultado["messages"][-1].content)

# --- Preguntas de reflexión (verificación rápida) ---
print("\n--- Tipo del último mensaje ---")
print(type(resultado["messages"][-1]))  # AIMessage
print(f"\nTotal de mensajes en el historial: {len(resultado['messages'])}")
